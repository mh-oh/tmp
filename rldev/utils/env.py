
import numpy as np
import torch as th

from collections import OrderedDict
from copy import deepcopy
from gymnasium import spaces
from typing import *

from rldev.utils.structure import recursive_map, instanceof, concatenate, dataclass


@dataclass
class Spec:
  shape: ...; dtype: ...


@dataclass
class Experience:
  u"""A transition or multiple transitions."""

  observation: np.ndarray
  action: np.ndarray
  reward: np.ndarray
  next_observation: np.ndarray
  done: np.ndarray


@dataclass
class DictExperience:
  u"""A transition or multiple transitions of 
  dictionary observations."""

  observation: OrderedDict
  action: np.ndarray
  reward: np.ndarray
  next_observation: OrderedDict
  done: np.ndarray

  def __len__(self):
    return len(self.reward)

  def get(self, index):
    
    def get(x):
      return x[index]
    
    observations = recursive_map(get, self.observation)
    actions = get(self.action)
    rewards = get(self.reward)
    next_observations = recursive_map(get, self.next_observation)
    dones = get(self.done)

    return DictExperience(observations,
                          actions,
                          rewards,
                          next_observations,
                          dones)
  
  def __getitem__(self, index):
    return self.get(index)

  @classmethod
  def concatenate(cls, inputs, axis):

    (observations,
     actions,
     rewards,
     next_observations,
     dones) = [], [], [], [], []
    for x in inputs:
      observations.append(x.observation)
      actions.append(x.action)
      rewards.append(x.reward)
      next_observations.append(x.next_observation)
      dones.append(x.done)

    def concat(inputs):
      return concatenate(inputs, axis=axis)
    return cls(concat(observations),
               concat(actions),
               concat(rewards),
               concat(next_observations),
               concat(dones))

    ...


def action_spec(space):
  if isinstance(space, spaces.Box):
    return Spec(space.shape, space.dtype)
  raise NotImplementedError()


def observation_spec(space):
  if isinstance(space, spaces.Box):
    return Spec(space.shape, space.dtype)
  if isinstance(space, spaces.Dict):
    return OrderedDict(
      (key, observation_spec(subspace)) 
        for (key, subspace) in space.spaces.items())
  raise NotImplementedError()


def observation_dim(space):
  if isinstance(space, spaces.Box):
    return space.shape[0]
  elif isinstance(space, spaces.Dict):
    return flatten_space(space).shape[0]
  raise NotImplementedError()


def action_dim(space):
  if isinstance(space, spaces.Box):
    return space.shape[0]
  raise NotImplementedError()


def container(size, spec, *, fill):

  if isinstance(spec, tuple):
    spec = Spec(*spec)

  if isinstance(spec, Spec):
    return np.full((*size, *spec.shape), fill, dtype=spec.dtype)
  elif isinstance(spec, OrderedDict):
    return recursive_map(
      lambda subspec: container(size, subspec, fill=fill), spec)
  
  raise NotImplementedError()


def flatten_space(space):
  
  if not isinstance(space, spaces.Dict):
    raise ValueError()
  if not isinstance(space.spaces, OrderedDict):
    raise AssertionError()

  low, high, dtype = [], [], []
  def append(x):
    low.append(x.low); high.append(x.high); dtype.append(x.dtype)

  for key, subspace in space.spaces.items():
    if isinstance(subspace, spaces.Dict):
      subspace = flatten_space(subspace)
    else:
      if not isinstance(subspace, spaces.Box):
        raise ValueError()
      if len(subspace.shape) > 1:
        subspace = spaces.Box(subspace.low.reshape(-1), 
                              subspace.high.reshape(-1), 
                              dtype=subspace.dtype)
    append(subspace)
  
  return spaces.Box(np.concatenate(low), np.concatenate(high),
                    dtype=np.result_type(*dtype))


def _reshape(x, shape):
  if isinstance(x, np.ndarray):
    return x.reshape(shape)
  if isinstance(x, th.Tensor):
    return x.reshape(*shape)
  raise NotImplementedError()


def _concatenate(xs, axis):
  if instanceof(*xs, type=np.ndarray):
    return np.concatenate(xs, axis=axis)
  if instanceof(*xs, type=th.Tensor):
    return th.cat(xs, dim=axis)
  raise NotImplementedError()


def flatten_observation(space: spaces.Dict, 
                        observation: Dict[str, Any]):

  if not isinstance(space.spaces, OrderedDict):
    raise AssertionError()

  xs = []
  for key, subspace in space.spaces.items():
    x = observation[key]
    if isinstance(subspace, spaces.Dict):
      x = flatten_observation(subspace, x)
    else:
      if not isinstance(subspace, spaces.Box):
        raise ValueError()
      if (ndims := len(subspace.shape)) > 1:
        if x.shape[-ndims:] != subspace.shape:
          raise ValueError()
        x = _reshape(x, (*x.shape[:-ndims], -1))
    xs.append(x)

  return _concatenate(xs, axis=-1)


def get_success_info(info: dict):
  sucess = info.get("success", None)
  if sucess is None:
    sucess = info.get("is_success", None)
  return sucess