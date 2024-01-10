

import gymnasium
from gymnasium.envs.mujoco.mujoco_env import DEFAULT_SIZE
from abc import *


class Env(gymnasium.Env, metaclass=ABCMeta):

  metadata = {
    "render_modes": ["human", "rgb_array", "depth_array"],
    "render_fps": 50}

  @abstractmethod
  def reset(self, *, seed=None, options=None):
    ...
  
  @abstractmethod
  def step(self, action):
    ...
  
  @abstractmethod
  def render(self):
    ...
  
  @property
  def render_mode(self):
    return "rgb_array"
  
  @property
  @abstractmethod
  def render_height(self):
    raise NotImplementedError()

  @property
  @abstractmethod
  def render_width(self):
    raise NotImplementedError()

  @property
  @abstractmethod
  def observation_space(self):
    ...
  
  @property
  @abstractmethod
  def action_space(self):
    ...
  
  @abstractmethod
  def compute_reward(self,
                     observation,
                     action,
                     next_observation,
                     info):
    ...
  
  @abstractmethod
  def compute_progress(self, observation):
    ...
  
  def discover_target(self, observation):
    ...
  
  def discover_object(self, observation):
    ...