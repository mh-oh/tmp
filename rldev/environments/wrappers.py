
import copy
import gymnasium
import numpy as np

from collections import OrderedDict
from gymnasium import spaces
from gymnasium.utils.step_api_compatibility import convert_to_done_step_api
from typing import *
from numpy.typing import *

from rldev.environments.core import Env


class BoxObservation(gymnasium.wrappers.FlattenObservation):

  def __init__(self, env):
    u""""""

    observation_space = env.observation_space
    if not isinstance(observation_space, spaces.Dict):
      raise ValueError(f"")
    self.dict_observation_space = observation_space

    super().__init__(env)

  def dict_observation(self, observation):
    return spaces.unflatten(
      self.dict_observation_space, observation)


class SuccessInfo(gymnasium.Wrapper):

  def __init__(self, env):
    super().__init__(env)

  def step(self, action):
    *step, info = super().step(action)
    if "is_success" in info:
      info["success"] = info["is_success"]
    return *step, info


class AddNoise(gymnasium.ObservationWrapper):
  
  def __init__(self, 
               env: Env, 
               low: Union[SupportsFloat, NDArray[Any]], 
               high: Union[SupportsFloat, NDArray[Any]], 
               shape: Union[Sequence[int], None] = None):
    super().__init__(env)
    
    space = copy.deepcopy(env.observation_space)
    if not isinstance(space, spaces.Dict):
      raise ValueError(
        f"observation space of 'env' should be 'spaces.Dict'")
    space["noise"] = spaces.Box(low, high, shape=shape)
    self.observation_space = space

  def observation(self, observation):
    
    if not isinstance(observation, (dict, OrderedDict)):
      raise AssertionError()
    observation["noise"] = self.observation_space["noise"].sample()
    return observation


class RGB(gymnasium.ObservationWrapper):
  
  def __init__(self, env: Env):
    super().__init__(env)
    if env.render_mode != "rgb_array":
      raise AssertionError()
    self.observation_space = (
      spaces.Box(low=-np.inf, high=np.inf,
                 shape=(env.render_height, env.render_width, 3)))
  
  def observation(self, observation):
    return self.render()


class GymApi(gymnasium.Wrapper):

  def __init__(self, env):
    super().__init__(env)
    self._seed = None
  
  def reset(self):
    seed = self._seed
    if seed is None:
      return self.env.reset()[0]
    observation, info = self.env.reset(seed=seed)
    self.env.observation_space.seed(seed=seed)
    self.env.action_space.seed(seed=seed)
    self._seed = None
    return observation
  
  def step(self, action):
    step_return = self.env.step(action)
    return convert_to_done_step_api(step_return)

  def seed(self, seed):
    self._seed = seed