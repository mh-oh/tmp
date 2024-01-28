
import numpy as np

from abc import *
from gymnasium import spaces
from overrides import overrides
from typing import *

from rldev.utils.env import observation_spec, action_spec, container, DictExperience
from rldev.utils.structure import recursive_map


class Buffer(metaclass=ABCMeta):

  def __init__(self,
               n_envs: int,
               capacity: int,
               observation_space: spaces.Space,
               action_space: spaces.Space):
    super().__init__()

    self._n_envs = n_envs
    self._capacity = capacity
    self._observation_spec = observation_spec(observation_space)
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


class DictBuffer(Buffer):
  u"""Replay buffer used in off-policy algorithms.
  This expects dictionary observations.
  """

  @overrides
  def __len__(self):
    return (self._capacity 
            if self._full else self._cursor) * self._n_envs

  def __init__(self,
               n_envs: int,
               capacity: int,
               observation_space: spaces.Dict,
               action_space: spaces.Dict,
               handle_timeouts: bool = True):
    super().__init__(n_envs, 
                     capacity, 
                     observation_space, 
                     action_space)
    self._capacity = max(capacity // n_envs, 1)

    self._observations = self._zeros(self._observation_spec)
    self._actions = self._zeros(self._action_spec)
    self._rewards = self._zeros(((), float))
    self._next_observations = self._zeros(self._observation_spec)
    self._dones = self._zeros(((), bool))

    self._handle_timeouts = handle_timeouts
    if handle_timeouts:
      self._timeouts = self._zeros(((), float))

  def _zeros(self, spec):
    return container(
      (self._capacity, self._n_envs), spec, fill=0)

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
    # if self._handle_timeouts:
    #   store(self._timeouts, np.array(
    #     [x.get("TimeLimit.truncated", False) for x in info]))

    def store(to, what):
      def fn(x, y):
        x[self._cursor, ...] = np.copy(y)
      recursive_map(fn, to, what)

    store(self._observations, observation)
    store(self._next_observations, next_observation)

    self._cursor += 1
    if self._cursor == self._capacity:
      self._full, self._cursor = True, 0

  def get(self, index):

    observations = self._recursive_get(self._observations, index)
    actions = self._actions[index].copy()
    rewards = self._rewards[index].copy()
    next_observations = self._recursive_get(self._next_observations, index)
    dones = self._dones[index].copy()

    return (observations,
            actions,
            rewards,
            next_observations,
            dones)

  @overrides
  def sample(self, size: int):
    upper_bound = self._capacity if self._full else self._cursor
    index = np.random.randint(0, upper_bound, size=size)

    n_envs = self._n_envs
    index = (index, 
             np.random.randint(
               0, high=n_envs, size=(len(index),)))

    (observations,
     actions,
     rewards,
     next_observations,
     dones) = self.get(index)
    rewards = rewards[index].reshape(size, 1).astype(np.float32)

    if self._handle_timeouts:
      dones = np.zeros_like(rewards, dtype=np.float32)

    return (observations,
            actions,
            rewards,
            next_observations,
            dones)


class EpisodicDictBuffer(DictBuffer):

  @overrides
  def __len__(self):
    return super().__len__()

  def __init__(self,
               n_envs: int,
               capacity: int,
               observation_space: spaces.Dict,
               action_space: spaces.Dict,
               handle_timeouts: bool = True):
    super().__init__(n_envs, 
                     capacity, 
                     observation_space, 
                     action_space,
                     handle_timeouts)

    capacity, n_envs = self._capacity, self._n_envs
    self._episode_cursor = np.zeros((n_envs,), dtype=int)
    self._episode_length = np.zeros((capacity, n_envs), dtype=int)
    self._episode_starts = np.zeros((capacity, n_envs), dtype=int)

  def add(self,
          observation,
          action,
          reward,
          next_observation,
          done,
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
                next_observation,
                done,
                info)

    for i in range(self._n_envs):
      if done[i]:
        s = self._episode_cursor[i]
        e = self._cursor
        if e < s:
          e += self._capacity
        index = np.arange(s, e) % self._capacity
        self._episode_length[index, i] = e - s
        self._episode_cursor[i] = self._cursor

  def episodes(self):
    
    capacity, n_envs = self._capacity, self._n_envs
    for i in range(n_envs):
      is_done_episode = self._episode_length[:, i] > 0
      starts = self._episode_starts[:, i][is_done_episode]
      length = self._episode_length[:, i][is_done_episode]
      for start, length in set(zip(starts, length)):
        yield (np.arange(start, start + length) % capacity, 
               np.ones((length,), dtype=int) * i)

  def get_episodes(self):
    
    indices, env_indices = [], []
    for index, env in self.episodes():
      indices.append(index); env_indices.append(env)
    
    indices = np.array(indices, dtype=int)
    env_indices = np.array(env_indices, dtype=int)
    assert indices.shape == env_indices.shape

    return DictExperience(
      *self.get((indices, env_indices)))

