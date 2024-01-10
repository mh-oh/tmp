
from typing import OrderedDict, Union
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


class Base(nn.Module):

  def __init__(self,
               observation_space: Union[spaces.Dict, spaces.Box],
               action_space: spaces.Box):
    super().__init__()
    self._observation_space = observation_space
    self._action_space = action_space
  
  @property
  @abstractmethod
  def optimizer(self):
    raise NotImplementedError()
  
  def predict(self,
              observation: OrderedDict,
              action: th.Tensor,
              **kwargs):
    ...
  
  def forward(self,
              observation: OrderedDict,
              action: th.Tensor,
              **kwargs):
    if not isinstance(action, th.Tensor):
      raise ValueError()
    if not isinstance(observation, (OrderedDict, dict)):
      raise ValueError()
    
    return self.predict(observation,
                        action,
                        **kwargs)


class FusionMLP(Base):

  DEFAULT_DIMS = [256, 256, 256]
  DEFAULT_ACTIVATIONS = ["leaky-relu", 
                         "leaky-relu", 
                         "leaky-relu", 
                         "tanh"]

  def __init__(self,
               observation_space: Union[spaces.Dict, spaces.Box],
               action_space: spaces.Box,
               fusion: int,
               dims: List[int] = DEFAULT_DIMS,
               activations: List[str] = DEFAULT_ACTIVATIONS):
    super().__init__(observation_space, action_space)

    odim = observation_dim(observation_space)
    adim = action_dim(action_space)
    def thunk():
      return MLP(dims=[odim + adim, *dims, 1],
                 activations=activations).float().to(thu.device())
    self._body = Fusion([thunk for _ in range(fusion)])

  def predict(self, 
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
    super().__init__(observation_space, action_space)
    self._dummy = nn.Linear(2, 1)
    self._optimizer = (
      th.optim.Adam(self._dummy.parameters(), lr=0.001))
  
  @property
  def optimizer(self):
    return self._optimizer

  def predict(self, observation, action):
    self._dummy(th.randn((2,)))
    target, object = (
      observation["desired_goal"], observation["achieved_goal"])
    return th.linalg.vector_norm(target - object, 
                                 dim=-1, keepdim=True)


class DistanceL2(Base):

  def __init__(self, 
               observation_space: Union[spaces.Dict, spaces.Box], 
               action_space: spaces.Box,
               *,
               common_dims: List[int] = [256, 256, 256],
               common_activations: List[str] = ["leaky-relu", "leaky-relu", "tanh"],
               projection_dims: List[int] = [128, 128],
               projection_activations: List[int] = ["tanh", "tanh"]):
    super().__init__(observation_space, action_space)
    
    odim = observation_dim(observation_space)
    adim = action_dim(action_space)

    self._common_body = (
      MLP(dims=[odim + adim, *common_dims],
          activations=common_activations).float().to(thu.device()))

    def projection():
      return MLP(dims=[common_dims[-1], *projection_dims],
                 activations=projection_activations).float().to(thu.device())
    self._psi = projection()
    self._phi = projection()

  def predict(self, 
              observation: OrderedDict, 
              action: th.Tensor, 
              **kwargs):
    observation = (
      flatten_observation(self._observation_space, 
                          observation))
    input = th.cat([observation, action], dim=-1)
    z = self._common_body(input)
    psi, phi = self._psi(z), self._phi(z)
    return th.linalg.vector_norm(psi - phi, 
                                 dim=-1, keepdim=True)

class Thunk(nn.Module):

  def __init__(self,
               input_dim,
               common_dims: List[int] = [256, 256, 256],
               common_activations: List[str] = ["leaky-relu", "leaky-relu", "tanh"],
               projection_dims: List[int] = [128, 128],
               projection_activations: List[int] = ["tanh", "tanh"]):
    super().__init__()
    self._common_body = (
      MLP(dims=[input_dim, *common_dims],
          activations=common_activations).float().to(thu.device()))
    def projection():
      return MLP(dims=[common_dims[-1], *projection_dims],
                 activations=projection_activations).float().to(thu.device())
    self._psi = projection()
    self._phi = projection()
  
  def forward(self, input):
    z = self._common_body(input)
    psi, phi = self._psi(z), self._phi(z)
    return th.linalg.vector_norm(psi - phi, 
                                 dim=-1, keepdim=True)


class FusionDistanceL2(Base):

  def __init__(self, 
               observation_space: Union[spaces.Dict, spaces.Box], 
               action_space: spaces.Box,
               *,
               fusion: int = 3,
               common_dims: List[int] = [256, 256, 256],
               common_activations: List[str] = ["leaky-relu", "leaky-relu", "tanh"],
               projection_dims: List[int] = [128, 128],
               projection_activations: List[int] = ["tanh", "tanh"]):
    super().__init__(observation_space, action_space)

    odim = observation_dim(observation_space)
    adim = action_dim(action_space)

    def thunk():
      return Thunk(odim + adim,
                   common_dims,
                   common_activations,
                   projection_dims,
                   projection_activations)
    self._body = Fusion([thunk for _ in range(fusion)])

  def predict(self, 
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
