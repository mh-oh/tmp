
import numpy as np
import torch as th

from abc import *
from collections import OrderedDict
from dataclasses import dataclass
from gym import spaces
from overrides import overrides
from typing import *

from rldev.utils import torch as thu
from rldev.utils.structure import *


@dataclass
class Spec:
  shape: ...; dtype: ...


@dataclass
class Step:
  u"""A transition or multiple transitions."""

  observation: np.ndarray
  action: np.ndarray
  reward: np.ndarray
  next_observation: np.ndarray
  done: np.ndarray


@dataclass
class DictStep:
  u"""A transition or multiple transitions of 
  dictionary observations."""

  observation: ArrDict
  action: np.ndarray
  reward: np.ndarray
  next_observation: ArrDict
  done: np.ndarray


def action_spec(space: spaces.Space):
  if isinstance(space, spaces.Box):
    return Spec(space.shape, space.dtype)
  raise NotImplementedError()


def observation_spec(space: spaces.Space):
  if isinstance(space, spaces.Box):
    return Spec(space.shape, space.dtype)
  if isinstance(space, spaces.Dict):
    return OrderedDict(
      (key, observation_spec(subspace)) 
        for (key, subspace) in space.spaces.items())
  raise NotImplementedError()


def copy(x):
  if isinstance(x, np.ndarray):
    return np.copy(x)
  elif isinstance(x, Mapping):
    def fn(x):
      if not isinstance(x, np.ndarray):
        raise AssertionError()
      return np.copy(x)
    return recursive_map(fn, x)
  else:
    raise AssertionError()


class Base(metaclass=ABCMeta):

  def __init__(self,
               n_envs: int,
               capacity: int,
               observation_space: spaces.Space,
               action_space: spaces.Space):
    super().__init__()

    self._n_envs = n_envs
    self._capacity = capacity
    self._observation_space = observation_space
    self._action_space = action_space
    self._observation_spec = observation_spec(observation_space)

    self._action_spec = action_spec(action_space)
    self._cursor = 0
    self._full = False

  def size(self):
    if self._full:
      return self._capacity
    return self._cursor

  @abstractmethod
  def add(self, *args, **kwargs):
    raise NotImplementedError()

  def extend(self, *args, **kwargs):
    for data in zip(*args):
      self.add(*data, **kwargs)

  def reset(self):
    self._cursor = 0
    self._full = False

  def sample(self, batch_size: int):
    upper_bound = self._capacity if self._full else self._cursor
    index = np.random.randint(0, upper_bound, size=batch_size)
    return self._get_samples(index)

  @abstractmethod
  def _get_samples(self, index):
    raise NotImplementedError()


class DictBuffer(Base):
  u"""Replay buffer used in off-policy algorithms.
  This expects dictionary observations.
  """

  def __init__(self,
               n_envs: int,
               capacity: int,
               observation_space: spaces.Dict,
               action_space: spaces.Dict):
    super().__init__(n_envs, 
                     capacity, 
                     observation_space, 
                     action_space)

    self._capacity = max(capacity // n_envs, 1)
    leading_shape = (self._capacity, self._n_envs)

    def container(spec):
      return np.zeros(
        (*leading_shape, *spec.shape), dtype=spec.dtype)
    def dict_container(spec):
      return ArrDict(
        recursive_map(container, spec), shape=leading_shape)

    self._observations = dict_container(self._observation_spec)
    self._next_observations = dict_container(self._observation_spec)
    self._actions = container(self._action_spec)
    self._rewards = container(Spec((), float))
    self._dones = container(Spec((), bool))
    self._infos = [None for _ in range(self._capacity)]

  def add(self,
          observation: Dict,
          next_observation: Dict,
          action: np.ndarray,
          reward: np.ndarray,
          done: np.ndarray,
          info: Dict[str, Any]):

    def store(to, what):
      to[self._cursor] = copy(what)

    store(self._observations, observation)
    store(self._next_observations, next_observation)
    store(self._actions, action.copy())
    store(self._rewards, reward.copy())
    store(self._dones, done.copy())
    self._infos[self._cursor] = info

    self._cursor += 1
    if self._cursor == self._capacity:
      self._full, self._cursor = True, 0
  
  @overrides
  def _get_samples(self, index):

    n_envs = self._n_envs
    index = (index, 
             np.random.randint(
               0, high=n_envs, size=(len(index),)))

    observations = self._observations[index]
    actions = self._actions[index]
    rewards = self._rewards[index]
    next_observations = self._next_observations[index]
    dones = self._dones[index]

    return Step(observations,
                actions,
                rewards,
                next_observations,
                dones)

