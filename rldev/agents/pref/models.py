
import numpy as np
import torch as th

from abc import *
from collections import OrderedDict
from gymnasium import spaces
from torch import nn
from typing import *

from rldev.utils import torch as thu
from rldev.utils.env import observation_dim, action_dim, flatten_observation
from rldev.utils.nn import Fusion, _MLP as MLP


DEFAULT_DIMS = [256, 256, 256]
DEFAULT_ACTIVATIONS = ["leaky-relu", "leaky-relu", "leaky-relu", "tanh"]


class Base(nn.Module):

  def __init__(self,
               observation_space,
               action_space):
    super().__init__()
    self._observation_space = observation_space
    self._action_space = action_space
  
  @property
  @abstractmethod
  def optimizer(self):
    raise NotImplementedError()


class FusionMLP(Base):
  
  def __init__(self,
               observation_space: Union[spaces.Dict, spaces.Box],
               action_space: spaces.Box,
               fusion: int,
               dims: List[int] = DEFAULT_DIMS,
               activations: List[str] = DEFAULT_ACTIVATIONS,
               lr: float = 3e-4):
    super().__init__(observation_space,
                     action_space)

    odim = observation_dim(observation_space)
    adim = action_dim(action_space)
    def thunk():
      return MLP(dims=[odim + adim, *dims, 1],
                 activations=activations).float().to(thu.device())
    self._body = Fusion([thunk for _ in range(fusion)])
    self._optimizer = (
      th.optim.Adam(self._body.parameters(), lr=lr))
  
  @property
  def optimizer(self):
    return self._optimizer

  def forward(self, 
              observation: OrderedDict, 
              action: th.Tensor,
              *, 
              member: int = -1, 
              reduce: str = "mean"):
    
    observation = (
      flatten_observation(self._observation_space, 
                          observation))
    input = th.cat([observation, action], dim=-1)
    if member != -1:
      return self._body[member](input)
    else:
      return self._body(input, reduce=reduce)


class TrueDistanceMLP(Base):

  def __init__(self,
               observation_space: Union[spaces.Dict, spaces.Box],
               action_space: spaces.Box):
    super().__init__(observation_space,
                     action_space)