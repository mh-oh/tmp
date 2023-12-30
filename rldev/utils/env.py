
import numpy as np

from collections import OrderedDict
from copy import deepcopy
from gym import spaces
from typing import *

from rldev.utils import gym_types
from rldev.utils.structure import AttrDict, recursive_map


def flatten_state(state, modalities=['observation', 'desired_goal']):
  #TODO: handle image modalities
  if isinstance(state, dict):
    return np.concatenate([state[m] for m in modalities], -1)
  return state


def discounted_sum(lst, discount):
  sum = 0
  gamma = 1
  for i in lst:
    sum += gamma*i
    gamma *= discount
  return sum


def debug_vectorized_experience(state, action, next_state, reward, done, info):
  """Gym returns an ambiguous "done" signal. VecEnv doesn't 
  let you fix it until now. See ReturnAndObsWrapper in env.py for where
  these info attributes are coming from."""
  experience = AttrDict(
    state = state,
    action = action,
    reward = reward,
    info = info
  )
  next_copy = deepcopy(next_state) # deepcopy handles dict states

  for idx in np.argwhere(done):
    i = idx[0]
    if isinstance(next_copy, np.ndarray):
      next_copy[i] = info[i].done_observation
    else:
      assert isinstance(next_copy, dict)
      for key in next_copy:
        next_copy[key][i] = info[i].done_observation[key]
  
  experience.next_state = next_copy
  experience.trajectory_over = done
  experience.done = np.array([info[i].terminal_state for i in range(len(done))], dtype=np.float32)
  experience.reset_state = next_state
  
  return next_state, experience


def dataclass(cls, /, **kwargs):

  from dataclasses import dataclass, fields
  class C(dataclass(cls, **kwargs)):
    __qualname__ = cls.__qualname__
    def __iter__(self):
      for field in fields(self):
        yield getattr(self, field.name)
  return C


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
      return x[index].copy()
    
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


def action_spec(space: gym_types.Space):
  if isinstance(space, gym_types.Box):
    return Spec(space.shape, space.dtype)
  raise NotImplementedError()


def observation_spec(space: gym_types.Space):
  if isinstance(space, gym_types.Box):
    return Spec(space.shape, space.dtype)
  if isinstance(space, gym_types.Dict):
    return OrderedDict(
      (key, observation_spec(subspace)) 
        for (key, subspace) in space.spaces.items())
  raise NotImplementedError()


def container(size, spec):
  return np.zeros((*size, *spec.shape), dtype=spec.dtype)


def dict_container(size, spec):
  return recursive_map(
    lambda spec: container(size, spec), spec)


def flatten_space(space: gym_types.Space):
  
  low, high, dtype = [], [], []
  def append(x):
    low.append(x.low); high.append(x.high); dtype.append(x.dtype)

  for key, subspace in space.spaces.items():
    if isinstance(subspace, gym_types.Dict):
      subspace = flatten_space(subspace)
    else:
      if not isinstance(subspace, gym_types.Box):
        raise ValueError()
      if len(subspace.shape) > 1:
        raise ValueError()
    append(subspace)
  
  return spaces.Box(low=np.concatenate(low),
                    high=np.concatenate(high),
                    dtype=np.result_type(*dtype))


def flatten_observation(space: gym_types.Dict, 
                        observation: Dict[str, Any]):

  xs = []
  for key, subspace in space.spaces.items():
    x = observation[key]
    if isinstance(subspace, gym_types.Dict):
      x = flatten_observation(subspace, x)
    else:
      if not isinstance(subspace, gym_types.Box):
        raise ValueError()
      if len(subspace.shape) > 1:
        raise ValueError()
    xs.append(x)
  return np.concatenate(xs, axis=-1)


def get_success_info(info: dict):
  sucess = info.get("success", None)
  if sucess is None:
    sucess = info.get("is_success", None)
  return sucess