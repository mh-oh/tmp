

from abc import *


class Env(metaclass=ABCMeta):

  @abstractmethod
  def reset(self):
    ...
  
  @abstractmethod
  def seed(self, seed):
    ...
  
  @abstractmethod
  def step(self, action):
    ...
  
  @abstractmethod
  def render(self):
    ...
  
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
  def compute_teacher_reward(self,
                             observation,
                             action,
                             next_observation,
                             info):
    ...
  
  def discover_target(self, observation):
    ...
  
  def discover_object(self, observation):
    ...