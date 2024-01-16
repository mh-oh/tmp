
import copy
import gymnasium
import numpy as np

from collections import OrderedDict
from gymnasium import spaces
from gymnasium.utils.step_api_compatibility import convert_to_done_step_api
from numpy.typing import *
from PIL import Image
from typing import *

from rldev.environments.core import Env


_registry = {}

def register(name):
  def decorator(cls):
    _registry[name] = cls; return cls
  return decorator

def get(name):
  return _registry[name]


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


@register("noise")
class AddNoise(gymnasium.ObservationWrapper):
  
  def __init__(self, 
               env: Env, 
               *,
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


@register("pixel")
class Pixel(gymnasium.ObservationWrapper):
  
  def __init__(self, 
               env: Env, 
               *,
               key: str = "pixels", 
               shape: Tuple[int, int] = None):
    super().__init__(env)
    if env.render_mode != "rgb_array":
      raise AssertionError()
    
    h, w = env.render_height, env.render_width
    if shape is not None:
      h, w = shape

    self.key = key
    self.shape = shape
    self.observation_space = spaces.Dict([(
      key,
      spaces.Box(0, 256, 
                 shape=(h, w, 3), dtype=np.uint8))])
  
  def observation(self, observation):
    shape, image = self.shape, self.render()
    if shape is not None:
      image = np.asarray(Image.fromarray(image).resize(shape))
    return OrderedDict([(self.key, image)])


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