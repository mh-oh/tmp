
from pathlib import Path
import numpy as np
import torch as th

from abc import *
from collections import OrderedDict
from gym import spaces
from overrides import overrides
from typing import *

from rldev.agents.core import Node, Agent
from rldev.utils import misc
from rldev.utils import torch as thu
from rldev.utils.env import *
from rldev.utils.structure import *


class Base(Node, metaclass=ABCMeta):

  def __init__(self,
               agent: Agent,
               n_envs: int,
               capacity: int,
               observation_space: spaces.Space,
               action_space: spaces.Space):
    super().__init__(agent)

    self._n_envs = n_envs
    self._capacity = capacity
    self._observation_space = observation_space
    self._observation_spec = observation_spec(observation_space)
    self._action_space = action_space
    self._action_spec = action_spec(action_space)
    
    self._cursor = 0
    self._full = False

  @abstractmethod
  def __len__(self):
    raise NotImplementedError()

  @abstractmethod
  def add(self, *args, **kwargs):
    raise NotImplementedError()

  def extend(self, *args, **kwargs):
    for data in zip(*args):
      self.add(*data, **kwargs)

  def reset(self):
    self._cursor = 0
    self._full = False

  @abstractmethod
  def sample(self, size: int):
    raise NotImplementedError()


class DictBuffer(Base):
  u"""Replay buffer used in off-policy algorithms.
  This expects dictionary observations.
  """

  @overrides
  def save(self, dir: Path): ...

  @overrides
  def load(self, dir: Path): ...

  @overrides
  def __len__(self):
    return (self._capacity if self._full else self._cursor) * self._n_envs

  def __init__(self,
               agent: Agent,
               n_envs: int,
               capacity: int,
               observation_space: spaces.Dict,
               action_space: spaces.Dict):
    super().__init__(agent,
                     n_envs, 
                     capacity, 
                     observation_space, 
                     action_space)
    self._capacity = max(capacity // n_envs, 1)

    self._observations = self._dict_container(self._observation_spec)
    self._actions = self._container(self._action_spec)
    self._rewards = self._container(Spec((), float))
    self._next_observations = self._dict_container(self._observation_spec)
    self._dones = self._container(Spec((), bool))
    self._infos = [None for _ in range(self._capacity)]

  def _container(self, spec):
    return container((self._capacity, self._n_envs), spec)

  def _dict_container(self, spec):
    return dict_container((self._capacity, self._n_envs), spec)

  def _recursive_get(self, x, index):
    return recursive_map(lambda x: x[index].copy(), x)

  def add(self,
          observation: Dict,
          action: np.ndarray,
          reward: np.ndarray,
          next_observation: Dict,
          done: np.ndarray,
          info: Dict[str, Any]):

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
  
  @overrides
  def sample(self, size: int):
    upper_bound = self._capacity if self._full else self._cursor
    index = np.random.randint(0, upper_bound, size=size)

    n_envs = self._n_envs
    index = (index, 
             np.random.randint(
               0, high=n_envs, size=(len(index),)))

    observations = self._recursive_get(self._observations, index)
    actions = self._actions[index].copy()
    rewards = self._rewards[index].reshape(size, 1).astype(np.float32)
    next_observations = self._recursive_get(self._next_observations, index)
    dones = self._dones[index].copy()

    if self._agent._config.get('never_done'):
      dones = np.zeros_like(rewards, dtype=np.float32)
    else:
      raise ValueError("Never done or first visit succ must be set in goal environments to use HER.")
    
    gammas = self._agent._config.gamma * (1. - dones)

    observations = np.concatenate([observations["observation"], observations["desired_goal"]], axis=-1)
    next_observations = np.concatenate([next_observations["observation"], next_observations["desired_goal"]], axis=-1)

    fn = self.agent._observation_normalizer
    if fn is not None:
      observations = fn(observations, update=False).astype(np.float32)
      next_observations = fn(next_observations, update=False).astype(np.float32)

    return (thu.torch(observations), thu.torch(actions),
          thu.torch(rewards), thu.torch(next_observations),
          thu.torch(gammas))


  def _process_experience(self, exp):
    self.add(exp.state,
             exp.action,
             exp.reward,
             exp.next_state,
             exp.done,
             {})


class EpisodicDictBuffer(DictBuffer):

  @overrides
  def save(self, dir: Path): ...

  @overrides
  def load(self, dir: Path): ...

  @overrides
  def __len__(self):
    return super().__len__()

  def __init__(self,
               agent: Agent,
               n_envs: int,
               capacity: int,
               observation_space: spaces.Dict,
               action_space: spaces.Dict):
    super().__init__(agent,
                     n_envs, 
                     capacity, 
                     observation_space, 
                     action_space)

    capacity, n_envs = self._capacity, self._n_envs
    self._episode_cursor = np.zeros((n_envs,), dtype=int)
    self._episode_length = np.zeros((capacity, n_envs), dtype=int)
    self._episode_starts = np.zeros((capacity, n_envs), dtype=int)

  def add(self,
          observation,
          action,
          reward,
          next_state,
          done,
          trajectory_over,
          info):

    # When the buffer is full, we rewrite on old episodes. 
    # When we start to rewrite on an old episode, we want the 
    # entire old episode to be deleted (and not only the transition 
    # on which we rewrite). To do this, we set the length of 
    # the old episode to 0, so it can't be sampled anymore.
    for i in range(self._n_envs):
      s = self._episode_starts[self._cursor, i]
      l = self._episode_length[self._cursor, i]
      if l > 0:
        index = np.arange(self._cursor, s + l) % self._capacity
        self._episode_length[index, i] = 0

    self._episode_starts[self._cursor, :] = np.copy(self._episode_cursor)

    super().add(observation,
                action,
                reward,
                next_state,
                done,
                info)

    for i in range(self._n_envs):
      if trajectory_over[i]:
        s = self._episode_cursor[i]
        e = self._cursor
        if e < s:
          e += self._capacity
        index = np.arange(s, e) % self._capacity
        self._episode_length[index, i] = e - s
        self._episode_cursor[i] = self._cursor

  def get(self, index):

    observations = self._recursive_get(self._observations, index)
    actions = self._actions[index].copy()
    rewards = self._rewards[index].copy()
    next_observations = self._recursive_get(self._next_observations, index)
    dones = self._dones[index].copy()

    return DictExperience(observations,
                          actions,
                          rewards,
                          next_observations,
                          dones)

  def episodes(self):
    
    capacity, n_envs = self._capacity, self._n_envs
    for i in range(n_envs):
      is_done_episode = self._episode_length[:, i] > 0
      starts = self._episode_starts[:, i][is_done_episode]
      length = self._episode_length[:, i][is_done_episode]
      for start, length in set(zip(starts, length)):
        yield (np.arange(start, 
                         start + length) % capacity, i)

  def get_episodes(self):
    return map(self.get, self.episodes())


class PEBBLEBuffer(DictBuffer):

  @overrides
  def save(self, dir: Path): ...

  @overrides
  def load(self, dir: Path): ...

  @overrides
  def __len__(self):
    return super().__len__()

  def __init__(self,
               agent: Agent,
               n_envs: int,
               capacity: int,
               observation_space: spaces.Dict,
               action_space: spaces.Dict):
    super().__init__(agent,
                     n_envs, 
                     capacity, 
                     observation_space, 
                     action_space)
    self._dones_no_max = self._container(Spec((), bool))

  def add(self,
          observation: Dict,
          action: np.ndarray,
          reward: np.ndarray,
          next_observation: Dict,
          done: np.ndarray,
          done_no_max: np.ndarray):

    self._dones_no_max[self._cursor] = np.copy(not done_no_max)
    super().add(observation,
                action,
                reward,
                next_observation,
                not done,
                {})

  @overrides
  def sample(self, size: int):

    upper_bound = self._capacity if self._full else self._cursor
    index = np.random.randint(0, upper_bound, size=size)

    n_envs = self._n_envs
    index = (index, 
             np.random.randint(
               0, high=n_envs, size=(len(index),)))

    observations = self._recursive_get(self._observations, index)
    actions = self._actions[index]
    rewards = self._rewards[index].reshape(size, 1).astype(np.float32)
    next_observations = self._recursive_get(self._next_observations, index)
    not_dones = self._dones[index].reshape(size, 1).astype(np.float32)
    not_dones_no_max = self._dones_no_max[index].reshape(size, 1).astype(np.float32)
    
    env = self.agent._env
    fun = env.to_box_observation
    observations = fun(observations)
    next_observations = fun(next_observations)

    return (thu.torch(observations), thu.torch(actions),
          thu.torch(rewards), thu.torch(next_observations),
          thu.torch(not_dones), thu.torch(not_dones_no_max))
  
  def _every_indices(self, ravel=True):

    index = self._capacity if self._full else self._cursor
    index = np.indices((index, self._n_envs))
    if ravel:
      index = tuple(map(np.ravel, index))
    return tuple(index)

  def sample_state_ent(self, size: int):
    
    (observations,
     actions,
     rewards,
     next_observations,
     not_dones,
     not_dones_no_max) = self.sample(size)

    env = self.agent._env
    fun = env.to_box_observation

    index = self._every_indices()
    every_observations = self._recursive_get(self._observations, index)
    every_observations = fun(every_observations)
    every_observations = thu.torch(every_observations)

    return (observations,
            every_observations,
            actions,
            rewards,
            next_observations,
            not_dones,
            not_dones_no_max)

  def relabel_rewards(self, predictor):

    env = self.agent._env

    index = self._every_indices()
    def batchify(*sequences, size):
      length = len(sequences[0])
      for s in sequences:
        if len(s) != length:
          raise ValueError("'sequences' should be of the same length")
      def maybe_tuple(gen):
        res = tuple(gen)
        if len(sequences) == 1:
          return res[0]
        return res
      for i in range(0, length, size):
        yield maybe_tuple(
          s[i : min(i + size, length)] for s in sequences)
    
    for batch in batchify(*index, size=256):

      observation = self._recursive_get(self._observations, batch)
      observation = env.to_box_observation(observation)
      action = self._actions[batch]

      input = np.concatenate([observation, action], axis=-1)      
      pred_reward = predictor.r_hat(input)
      self._rewards[batch] = thu.numpy(pred_reward)[..., 0]


