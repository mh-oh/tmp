
import gym
import numpy as np
import pickle

from collections import OrderedDict
from gym import spaces
from overrides import overrides
from pathlib import Path

from rldev.agents.core import Node, Agent
from rldev.buffers.basic import DictBuffer, DictExperience
from rldev.utils import torch as ptu
from rldev.utils.structure import recursive_map

class HindsightBuffer(DictBuffer):

  def __init__(self, 
               agent: Agent, 
               n_envs: int, 
               capacity: int, 
               observation_space: spaces.Dict, 
               action_space: spaces.Dict,
               mode: str):
    super().__init__(agent, 
                     n_envs, 
                     capacity, 
                     observation_space, 
                     action_space)

    capacity, n_envs = self._capacity, self._n_envs
    self._episode_cursor = np.zeros((n_envs,), dtype=int)
    self._episode_length = np.zeros((capacity, n_envs), dtype=int)
    self._episode_starts = np.zeros((capacity, n_envs), dtype=int)

    self._fut, self._act, self._ach, self._beh = parse_hindsight_mode(mode)

  def _process_experience(self, exp):

    for i in range(self._n_envs):
      s = self._episode_starts[self._cursor, i]
      l = self._episode_length[self._cursor, i]
      if l > 0:
        index = np.arange(self._cursor, s + l) % self._capacity
        self._episode_length[index, i] = 0

    self._episode_starts[self._cursor, :] = np.copy(self._episode_cursor)

    super().add(exp.state,
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
        index = np.arange(s, e) % self._capacity
        self._episode_length[index, i] = e - s
        self._episode_cursor[i] = self._cursor

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

  @overrides
  def sample(self, size: int):

    is_episode = self._episode_length > 0
    if not np.any(is_episode):
      raise ValueError(f"")

    episode_index = np.flatnonzero(is_episode)
    index = np.random.choice(episode_index, size=size, replace=True)

    if self.agent.env_steps > self.agent.config.future_warm_up:
      fut_batch_size, act_batch_size, ach_batch_size, beh_batch_size, real_batch_size = np.random.multinomial(
          size, [self._fut, self._act, self._ach, self._beh, 1.])
    else:
      fut_batch_size, act_batch_size, ach_batch_size, beh_batch_size, real_batch_size = size, 0, 0, 0, 0
    
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

    info = {"next_observation": next_observations,
            "action": actions}
    rewards = self._agent._env.compute_reward(
      next_observations["achieved_goal"], observations["desired_goal"], info).reshape(size, 1).astype(np.float32)

    if self._agent._config.get('never_done'):
      dones = np.zeros_like(rewards, dtype=np.float32)
    elif self._agent._config.get('first_visit_succ'):
      dones = np.round(rewards + 1.)
    else:
      raise ValueError("Never done or first visit succ must be set in goal environments to use HER.")
    
    gammas = self._agent._config.gamma * (1. - dones)

    observations = np.concatenate([observations["observation"], observations["desired_goal"]], axis=-1)
    next_observations = np.concatenate([next_observations["observation"], next_observations["desired_goal"]], axis=-1)
    
    fn = self.agent._observation_normalizer
    if fn is not None:
      observations = fn(observations, update=False).astype(np.float32)
      next_observations = fn(next_observations, update=False).astype(np.float32)

    return (ptu.torch(observations), ptu.torch(actions),
          ptu.torch(rewards), ptu.torch(next_observations),
          ptu.torch(gammas))

  @overrides
  def __len__(self):
    return (self._capacity if self._full else self._cursor) * self._n_envs

  @overrides
  def save(self, dir: Path):
    return
    dir.mkdir(parents=True, exist_ok=True)
    state = self._buffer._get_state()
    with open(dir / "_buffer.pkl", "wb") as fout:
      pickle.dump(state, fout)

  @overrides
  def load(self, dir: Path):
    return
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
