
import torch as th

from abc import *
from gymnasium import spaces
from typing import *

from torch import nn

from rldev.utils.env import flatten_observation, flatten_space


class Extractor(nn.Module):

  def __init__(self, 
               observation_space: spaces.Space):
    super().__init__()
    self._observation_space = observation_space

  @property
  def observation_space(self):
    return self._observation_space

  @property
  def feature_space(self):
    ...


class Flatten(Extractor):
  
  def __init__(self, 
               observation_space: spaces.Dict):
    super().__init__(observation_space)

  def forward(self, 
              observation: Dict[str, Any]):
    return flatten_observation(self.observation_space,
                               observation)

  @property
  def feature_space(self):
    return flatten_space(self.observation_space)

