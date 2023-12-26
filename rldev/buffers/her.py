
import gym
import numpy as np
import pickle

from collections import OrderedDict
from gym import spaces
from overrides import overrides
from pathlib import Path

from rldev.agents.core import Node
from rldev.buffers.core.shared_buffer import SharedMemoryTrajectoryBuffer as Buffer
from rldev.utils import torch as ptu
from rldev.utils.env import flatten_state

from rldev.buffers.basic import *


class OnlineHERBuffer(Node):

  def __init__(self, agent):
    super().__init__(agent)

    self._size = None
    self._goal_shape = None
    self._buffer = None
    self.save_buffer = None # can be manually set to save this replay buffer irrespective of config
    self._modalities = ['observation']
    self._goal_modalities = ['desired_goal']
    
    self._size = self._agent._config.replay_size

    env = self._agent._env

    if type(env.observation_space) == gym.spaces.Dict:
      if env.goal_env:
        self._goal_modalities = [m for m in self._agent._config.goal_modalities]
        self._goal_shape = (env.goal_dim,)
      state_shape = (env.state_dim,)
      self._modalities = [m for m in self._agent._config.modalities]
    else:
      raise
      state_shape = env.observation_space.shape

    items = [("state", state_shape),
             ("action", env.action_space.shape), ("reward", (1,)),
             ("next_state", state_shape), ("done", (1,))]

    if self._goal_shape is not None:
      items += [("previous_ag", self._goal_shape), # for reward shaping
                ("ag", self._goal_shape), # achieved goal
                ("bg", self._goal_shape), # behavioral goal (i.e., intrinsic if curious agent)
                ("dg", self._goal_shape)] # desired goal (even if ignored behaviorally)

    if isinstance(env.observation_space, spaces.Dict):
      observation_dim = 0
      self._subspace_keys = []
      for key, subspace in env.observation_space.spaces.items():
        if not isinstance(subspace, spaces.Box):
          raise AssertionError()
        dim, = subspace.shape
        observation_dim += dim
        self._subspace_keys.append(key)
      print(self._subspace_keys, observation_dim)
      def dict_observation(x):
        subx = []
        i = 0
        for key in self._subspace_keys:
          dim, = env.observation_space.spaces[key].shape
          subx.append((key, x[..., i:i+dim]))
          i += dim
        return OrderedDict(subx)
      self._dict_observation = dict_observation

      items += [("original_observation", (observation_dim,))]

    self._buffer = Buffer(self._size, items)
    self._subbuffers = [[] for _ in range(self._agent._env.num_envs)]
    self._n_envs = self._agent._env.num_envs

    # HER mode can differ if demo or normal replay buffer
    # if 'demo' in self.module_name:
    #   self._fut, self._act, self._ach, self._beh = parse_hindsight_mode(self._agent._config.demo_her)
    # else:
    self._fut, self._act, self._ach, self._beh = parse_hindsight_mode(self._agent._config.her)


    ####################################
    n_envs = env.num_envs
    capacity = self._size
    observation_space = env.observation_space
    action_space = env.action_space

    self._capacity = capacity
    self._observation_space = observation_space
    self._observation_spec = observation_spec(observation_space)
    self._action_space = action_space
    self._action_spec = action_spec(action_space)

    print("observation_spec:")
    print(self._observation_spec)
    print("action_spec:")
    print(self._action_spec)

    self._capacity = max(capacity // n_envs, 1)
    leading_shape = (self._capacity, self._n_envs)

    def container(spec):
      return np.zeros(
        (*leading_shape, *spec.shape), dtype=spec.dtype)
    def dict_container(spec):
      return recursive_map(container, spec)

    self._observations = dict_container(self._observation_spec)
    self._actions = container(self._action_spec)
    self._rewards = container(Spec((), float))
    self._next_observations = dict_container(self._observation_spec)
    self._dones = container(Spec((), bool))
    self._infos = [None for _ in range(self._capacity)]

    self._cursor = 0
    self._full = False

    self._episode_cursor = np.zeros((n_envs,), dtype=int)
    self._episode_length = np.zeros((self._capacity, n_envs), dtype=int)
    self._episode_starts = np.zeros((self._capacity, n_envs), dtype=int)
    ###################################


  def _len(self):
    return len(self._infos)

  def _add(self,
           observation: Dict,
           action: np.ndarray,
           reward: np.ndarray,
           next_observation: Dict,
           done: np.ndarray,
           info: Dict[str, Any]):

    assert (self._buffer.BUFF.buffer_bg.data != self._buffer.BUFF.buffer_dg.data).sum() <= 0

    def store(to, what):
      to[self._cursor] = np.copy(what)

    store(self._actions, action)
    store(self._rewards, reward)
    store(self._dones, done)

    def store(to, what):
      def fn(x, y):
        x[self._cursor, ...] = np.copy(y)
      recursive_map(fn, to, what)

    store(self._observations, observation)
    store(self._next_observations, next_observation)

    self._infos[self._cursor] = info

    self._cursor += 1
    if self._cursor == self._capacity:
      self._full, self._cursor = True, 0

  def _process_experience(self, exp):

    for i in range(self._n_envs):
      s = self._episode_starts[self._cursor, i]
      l = self._episode_length[self._cursor, i]
      if l > 0:
        index = np.arange(self._cursor, s + l) % self._capacity
        self._episode_length[index, i] = 0

    self._episode_starts[self._cursor, :] = np.copy(self._episode_cursor)

    self._add(exp.state,
              exp.action,
              exp.reward,
              exp.next_state,
              exp.done,
              {})

    for i in range(self._n_envs):
      if exp.trajectory_over[i]:
        s = self._episode_cursor[i]
        e = self._cursor
        if e < s:
          e += self._capacity
        index = np.arange(s, e) % self._cursor
        self._episode_length[index, i] = e - s
        self._episode_cursor[i] = self._cursor

    # if getattr(self._agent, 'logger'):
    self._agent._logger.add_tabular('Replay buffer size', len(self._buffer))
    done = np.expand_dims(exp.done, 1)  # format for replay buffer
    reward = np.expand_dims(exp.reward, 1)  # format for replay buffer

    action = exp.action

    if self._goal_shape:
      state = flatten_state(exp.state, self._modalities)
      next_state = flatten_state(exp.next_state, self._modalities)
      if hasattr(self._agent, 'achieved_goal'):
        raise
        previous_achieved = self._agent._achieved_goal(exp.state)
        achieved = self._agent._achieved_goal(exp.next_state)
      else:
        previous_achieved = exp.state['achieved_goal']
        achieved = exp.next_state['achieved_goal']
      desired = flatten_state(exp.state, self._goal_modalities)
      if hasattr(self._agent, 'ag_curiosity') and self._agent.ag_curiosity.current_goals is not None:
        raise
        behavioral = self._agent.ag_curiosity.current_goals
        # recompute online reward
        reward = self._agent._env.compute_reward(achieved, behavioral, {'s':state, 'a':action, 'ns':next_state}).reshape(-1, 1)
      else:
        behavioral = desired
      observation = np.concatenate([exp.state[key] for key in self._subspace_keys], axis=-1)
      for i in range(self._n_envs):
        self._subbuffers[i].append([
            state[i], action[i], reward[i], next_state[i], done[i], previous_achieved[i], achieved[i],
            behavioral[i], desired[i], observation[i]
        ])
    else:
      raise
      state = exp.state
      next_state = exp.next_state
      for i in range(self._n_envs):
        self._subbuffers[i].append(
            [state[i], action[i], reward[i], next_state[i], done[i]])

    for i in range(self._n_envs):
      if exp.trajectory_over[i]:
        trajectory = [np.stack(a) for a in zip(*self._subbuffers[i])]
        self._buffer.add_trajectory(*trajectory)
        self._subbuffers[i] = []

  def none(self, index):

    index = np.unravel_index(index, self._episode_length.shape)

    def get(x, index):
      return recursive_map(lambda x: x[index], x)

    observations = get(self._observations, index)
    actions = self._actions[index]
    rewards = self._rewards[index]
    next_observations = get(self._next_observations, index)
    dones = self._dones[index]

    return DictExperience(observations,
                          actions,
                          rewards,
                          next_observations,
                          dones)

  def real(self, index):

    (observations, 
     actions, 
     rewards, 
     next_observations, 
     dones) = self.none(index)
    
    new_goals = np.copy(observations.get("desired_goal")) #############

    observations["desired_goal"] = new_goals
    next_observations["desired_goal"] = new_goals
    
    return DictExperience(observations,
                          actions,
                          rewards,
                          next_observations,
                          dones)

  def future(self, index):

    (observations, 
     actions, 
     rewards, 
     next_observations, 
     dones) = self.none(index)

    index = np.unravel_index(index, self._episode_length.shape)
    batch_index, env_index = index

    episode_starts = self._episode_starts[index]
    episode_length = self._episode_length[index]
    future_index = np.random.randint(batch_index, episode_starts + episode_length) % self._capacity
    new_goals = self._next_observations["achieved_goal"][future_index, env_index]

    observations["desired_goal"] = new_goals
    next_observations["desired_goal"] = new_goals
    
    return DictExperience(observations,
                          actions,
                          rewards,
                          next_observations,
                          dones)
  
  def desired(self, index):
    
    (observations, 
     actions, 
     rewards, 
     next_observations, 
     dones) = self.none(index)
    
    batch_index = np.random.choice(self._cursor, size=index.shape)
    env_index = np.random.choice(self._n_envs, size=index.shape)
    new_goals = self._observations["desired_goal"][batch_index, env_index]

    observations["desired_goal"] = new_goals
    next_observations["desired_goal"] = new_goals

    return DictExperience(observations,
                          actions,
                          rewards,
                          next_observations,
                          dones)

  def achieved(self, index):
    
    (observations, 
     actions, 
     rewards, 
     next_observations, 
     dones) = self.none(index)
    
    batch_index = np.random.choice(self._cursor, size=index.shape)
    env_index = np.random.choice(self._n_envs, size=index.shape)
    new_goals = self._next_observations["achieved_goal"][batch_index, env_index]

    observations["desired_goal"] = new_goals
    next_observations["desired_goal"] = new_goals

    return DictExperience(observations,
                          actions,
                          rewards,
                          next_observations,
                          dones)

  def behavior(self, index):
    
    (observations, 
     actions, 
     rewards, 
     next_observations, 
     dones) = self.none(index)
    
    batch_index = np.random.choice(self._cursor, size=index.shape)
    env_index = np.random.choice(self._n_envs, size=index.shape)
    new_goals = self._observations["desired_goal"][batch_index, env_index] ##################

    observations["desired_goal"] = new_goals
    next_observations["desired_goal"] = new_goals

    return DictExperience(observations,
                          actions,
                          rewards,
                          next_observations,
                          dones)

  def sample(self, batch_size, to_torch=True):

    is_episode = self._episode_length > 0
    if not np.any(is_episode):
      raise ValueError(f"")

    episode_index = np.flatnonzero(is_episode)
    index = np.random.choice(episode_index, size=batch_size, replace=True)

    if hasattr(self._agent, 'prioritized_replay'):
      raise
      batch_idxs = self._agent.prioritized_replay(batch_size)
    else:
      batch_idxs = np.random.randint(self._buffer.size, size=batch_size)

    if self._goal_shape:
      # if "demo" in self.module_name:
      #   has_config_her = self._agent._config.get('demo_her')
      # else:
      has_config_her = self._agent._config.get('her')
      
      if has_config_her:

        if self._agent._config.env_steps > self._agent._config.future_warm_up:
          fut_batch_size, act_batch_size, ach_batch_size, beh_batch_size, real_batch_size = np.random.multinomial(
              batch_size, [self._fut, self._act, self._ach, self._beh, 1.])
        else:
          fut_batch_size, act_batch_size, ach_batch_size, beh_batch_size, real_batch_size = batch_size, 0, 0, 0, 0

        fut_idxs, act_idxs, ach_idxs, beh_idxs, real_idxs = np.array_split(batch_idxs, 
          np.cumsum([fut_batch_size, act_batch_size, ach_batch_size, beh_batch_size]))
        
        future, actual, achieved, behavior, real = np.array_split(index, 
          np.cumsum([fut_batch_size, act_batch_size, ach_batch_size, beh_batch_size]))

        real = self.real((real))
        future = self.future((future))
        actual = self.desired((actual))
        achieved = self.achieved((achieved))
        behavior = self.behavior((behavior))

        def concatenate(*args):
          return np.concatenate(args, axis=0)

        observations = recursive_map(
          concatenate, *(real.observation, 
                         future.observation, 
                         actual.observation, 
                         achieved.observation, 
                         behavior.observation))
        next_observations = recursive_map(
          concatenate, *(real.next_observation, 
                         future.next_observation, 
                         actual.next_observation, 
                         achieved.next_observation, 
                         behavior.next_observation))
        actions = concatenate(real.action,
                              future.action,
                              actual.action,
                              achieved.action,
                              behavior.action)

        rewards = self._agent._env.compute_reward(
          next_observations["achieved_goal"], observations["desired_goal"], {}).reshape(-1, 1).astype(np.float32)

        if self._agent._config.get('never_done'):
          dones = np.zeros_like(rewards, dtype=np.float32)
        elif self._agent._config.get('first_visit_succ'):
          dones = np.round(rewards + 1.)
        else:
          raise ValueError("Never done or first visit succ must be set in goal environments to use HER.")
        
        gammas = self._agent._config.gamma * (1. - dones)

        observations = np.concatenate([observations["observation"], observations["desired_goal"]], axis=-1)
        next_observations = np.concatenate([next_observations["observation"], next_observations["desired_goal"]], axis=-1)
        if self._agent._observation_normalizer is not None:
          fn = self.agent._observation_normalizer
          observations = fn(observations, update=False).astype(np.float32)
          next_observations = fn(next_observations, update=False).astype(np.float32)

        B, _ = observations.shape
        if batch_size != B:
          raise AssertionError(f"{batch_size} != {B}")
        B, _ = actions.shape
        if batch_size != B:
          raise AssertionError(f"{batch_size} != {B}")
        if to_torch:
          return (ptu.torch(observations), ptu.torch(actions),
                ptu.torch(rewards), ptu.torch(next_observations),
                ptu.torch(gammas))
        else:
          return (observations, actions, rewards, next_observations, gammas)

        print(observations["observation"].shape)

        exit(0)
        print(future.observation["observation"].shape)
        x = concatenate((future.observation, future.observation), axis=0)
        print(x.get("observation").shape)
        print(x.shape)
        observations = concatenate(
          (real.observation, 
           future.observation, 
           actual.observation, 
           achieved.observation, 
           behavior.observation), axis=0)
        actions = np.concatenate(
          (real.action, 
           future.action, 
           actual.action, 
           achieved.action, 
           behavior.action), axis=0)
        rewards = np.concatenate(
          (real.reward, 
           future.reward, 
           actual.reward, 
           achieved.reward, 
           behavior.reward), axis=0)
        next_observations = concatenate(
          (real.next_observation, 
           future.next_observation, 
           actual.next_observation, 
           achieved.observation, 
           behavior.next_observation), axis=0)
        dones = np.concatenate(
          (real.done, 
           future.done, 
           actual.done, 
           achieved.done, 
           behavior.done), axis=0)
        
        rewards = self._agent._env.compute_reward(
          next_observations.get("achieved_goal"), observations.get("desired_goal"), {}).reshape(-1, 1).astype(np.float32)

        if self._agent._config.get('never_done'):
          dones = np.zeros_like(rewards, dtype=np.float32)
        elif self._agent._config.get('first_visit_succ'):
          dones = np.round(rewards + 1.)
        else:
          raise ValueError("Never done or first visit succ must be set in goal environments to use HER.")
        
        gammas = self._agent._config.gamma * (1. - dones)

        if to_torch:
          return (ptu.torch(observations), ptu.torch(actions),
                ptu.torch(rewards), ptu.torch(next_observations),
                ptu.torch(gammas))
        else:
          return (observations, actions, rewards, next_observations, gammas)

        exit(0)

        # print(fut_idxs.shape, future.shape)
        # print(act_idxs.shape)
        # print(ach_idxs.shape)

        # Sample the real batch (i.e., goals = behavioral goals)
        states, actions, rewards, next_states, dones, previous_ags, ags, goals, _, original =\
            self._buffer.sample(real_batch_size, batch_idxs=real_idxs)

        # Sample the future batch
        states_fut, actions_fut, _, next_states_fut, dones_fut, previous_ags_fut, ags_fut, _, _, original_fut, goals_fut =\
          self._buffer.sample_future(fut_batch_size, batch_idxs=fut_idxs)
        # Sample the actual batch
        # state, action, reward, next_state, done, previous_ag, ag, bg, dg
        states_act, actions_act, _, next_states_act, dones_act, previous_ags_act, ags_act, _, __dg, original_act, goals_act =\
          self._buffer.sample_from_goal_buffer('dg', act_batch_size, batch_idxs=act_idxs)
        # assert (__dg != goals_act).sum() <= 0

        # Sample the achieved batch
        states_ach, actions_ach, _, next_states_ach, dones_ach, previous_ags_ach, ags_ach, _, _, original_ach, goals_ach =\
          self._buffer.sample_from_goal_buffer('ag', ach_batch_size, batch_idxs=ach_idxs)

        # Sample the behavioral batch
        states_beh, actions_beh, _, next_states_beh, dones_beh, previous_ags_beh, ags_beh, _, _, original_beh, goals_beh =\
          self._buffer.sample_from_goal_buffer('bg', beh_batch_size, batch_idxs=beh_idxs)

        # Concatenate the five
        states = np.concatenate([states, states_fut, states_act, states_ach, states_beh], 0)
        actions = np.concatenate([actions, actions_fut, actions_act, actions_ach, actions_beh], 0)
        ags = np.concatenate([ags, ags_fut, ags_act, ags_ach, ags_beh], 0)
        goals = np.concatenate([goals, goals_fut, goals_act, goals_ach, goals_beh], 0)
        next_states = np.concatenate([next_states, next_states_fut, next_states_act, next_states_ach, next_states_beh], 0)
        original = np.concatenate([original, original_fut, original_act, original_ach, original_beh], axis=0)

        # Recompute reward online
        if hasattr(self._agent, 'goal_reward'):
          raise
          rewards = self._agent.goal_reward(ags, goals, {'s':states, 'a':actions, 'ns':next_states}).reshape(-1, 1).astype(np.float32)
        else:
          dict_obs = self._dict_observation(original)
          dict_obs["achieved_goal"] = ags
          dict_obs["desired_goal"] = goals
          rewards = self._agent._env.compute_reward(ags, goals, {'s':states, 'a':actions, 'ns':next_states, "dict": dict_obs}).reshape(-1, 1).astype(np.float32)

        if self._agent._config.get('never_done'):
          dones = np.zeros_like(rewards, dtype=np.float32)
        elif self._agent._config.get('first_visit_succ'):
          dones = np.round(rewards + 1.)
        else:
          raise ValueError("Never done or first visit succ must be set in goal environments to use HER.")
          dones = np.concatenate([dones, dones_fut, dones_act, dones_ach, dones_beh], 0)

        if self._agent._config.sparse_reward_shaping:
          raise
          previous_ags = np.concatenate([previous_ags, previous_ags_fut, previous_ags_act, previous_ags_ach, previous_ags_beh], 0)
          previous_phi = -np.linalg.norm(previous_ags - goals, axis=1, keepdims=True)
          current_phi  = -np.linalg.norm(ags - goals, axis=1, keepdims=True)
          rewards_F = self._agent._config.gamma * current_phi - previous_phi
          rewards += self._agent._config.sparse_reward_shaping * rewards_F

      else:
        raise
        # Uses the original desired goals
        states, actions, rewards, next_states, dones, _ , _, _, goals =\
                                                    self._buffer.sample(batch_size, batch_idxs=batch_idxs)

      if self._agent._config.slot_based_state:
        raise
        # TODO: For now, we flatten according to config.slot_state_dims
        I, J = self._agent._config.slot_state_dims
        states = np.concatenate((states[:, I, J], goals), -1)
        next_states = np.concatenate((next_states[:, I, J], goals), -1)
      else:
        states = np.concatenate((states, goals), -1)
        next_states = np.concatenate((next_states, goals), -1)
      gammas = self._agent._config.gamma * (1.-dones)

    elif self._agent._config.get('n_step_returns') and self._agent._config.n_step_returns > 1:
      raise
      states, actions, rewards, next_states, dones = self._buffer.sample_n_step_transitions(
        batch_size, self._agent._config.n_step_returns, self._agent._config.gamma, batch_idxs=batch_idxs
      )
      gammas = self._agent._config.gamma**self._agent._config.n_step_returns * (1.-dones)

    else:
      raise
      states, actions, rewards, next_states, dones = self._buffer.sample(
          batch_size, batch_idxs=batch_idxs)
      gammas = self._agent._config.gamma * (1.-dones)

    if self._agent._observation_normalizer is not None:
      states = self._agent._observation_normalizer(states, update=False).astype(np.float32)
      next_states = self._agent._observation_normalizer(
          next_states, update=False).astype(np.float32)
    
    if to_torch:
      return (ptu.torch(states), ptu.torch(actions),
            ptu.torch(rewards), ptu.torch(next_states),
            ptu.torch(gammas))
    else:
      return (states, actions, rewards, next_states, gammas)

  def __len__(self):
    return len(self._buffer)

  @overrides
  def save(self, dir: Path):
    dir.mkdir(parents=True, exist_ok=True)
    state = self._buffer._get_state()
    with open(dir / "_buffer.pkl", "wb") as fout:
      pickle.dump(state, fout)

  @overrides
  def load(self, dir: Path):
    with open(dir / "_buffer.pkl", "rb") as fin:
      state = pickle.load(fin)
    self._buffer._set_state(state)

def parse_hindsight_mode(hindsight_mode : str):
  if 'future_' in hindsight_mode:
    _, fut = hindsight_mode.split('_')
    fut = float(fut) / (1. + float(fut))
    act = 0.
    ach = 0.
    beh = 0.
  elif 'futureactual_' in hindsight_mode:
    _, fut, act = hindsight_mode.split('_')
    non_hindsight_frac = 1. / (1. + float(fut) + float(act))
    fut = float(fut) * non_hindsight_frac
    act = float(act) * non_hindsight_frac
    ach = 0.
    beh = 0.
  elif 'futureachieved_' in hindsight_mode:
    _, fut, ach = hindsight_mode.split('_')
    non_hindsight_frac = 1. / (1. + float(fut) + float(ach))
    fut = float(fut) * non_hindsight_frac
    act = 0.
    ach = float(ach) * non_hindsight_frac
    beh = 0.
  elif 'rfaa_' in hindsight_mode:
    _, real, fut, act, ach = hindsight_mode.split('_')
    denom = (float(real) + float(fut) + float(act) + float(ach))
    fut = float(fut) / denom
    act = float(act) / denom
    ach = float(ach) / denom
    beh = 0.
  elif 'rfaab_' in hindsight_mode:
    _, real, fut, act, ach, beh = hindsight_mode.split('_')
    denom = (float(real) + float(fut) + float(act) + float(ach) + float(beh))
    fut = float(fut) / denom
    act = float(act) / denom
    ach = float(ach) / denom
    beh = float(beh) / denom
  else:
    fut = 0.
    act = 0.
    ach = 0.
    beh = 0.

  return fut, act, ach, beh
