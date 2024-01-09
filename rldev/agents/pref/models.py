
import torch as th

from gymnasium import spaces
from torch import nn
from typing import *

from rldev.utils import torch as thu
from rldev.utils.env import observation_dim, action_dim
from rldev.utils.nn import Fusion, _MLP as MLP


DEFAULT_DIMS = [256, 256, 256]
DEFAULT_ACTIVATIONS = ["leaky-relu", "leaky-relu", "leaky-relu", "tanh"]


class FusionMLP(Fusion):
  
  def __init__(self,
               observation_space: Union[spaces.Dict, spaces.Box],
               action_space: spaces.Box,
               fusion: int,
               dims: List[int] = DEFAULT_DIMS,
               activations: List[str] = DEFAULT_ACTIVATIONS,
               lr: float = 3e-4):

    odim = observation_dim(observation_space)
    adim = action_dim(action_space)
    def thunk():
      return MLP(dims=[odim + adim, *dims, 1],
                 activations=activations).float().to(thu.device())
    super().__init__([thunk for _ in range(fusion)])

    self._optimizer = th.optim.Adam(self.parameters(), lr=lr)
  
  @property
  def optimizer(self):
    return self._optimizer


class TrueDistanceMLP(nn.Module):

  def __init__(self,
               observation_space: Union[spaces.Dict, spaces.Box],
               action_space: spaces.Box):
    ...