
import numpy as np

from collections import OrderedDict
from copy import deepcopy
from gym import spaces

from rldev.utils.structure import AttrDict


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

