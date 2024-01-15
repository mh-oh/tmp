
from typing import OrderedDict, Union
import numpy as np
import torch as th

from abc import *
from collections import OrderedDict
from gymnasium import spaces
from torch import nn
from typing import *

from rldev.utils import torch as thu
from rldev.utils.env import observation_dim, action_dim
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
              observation: th.Tensor,
              action: th.Tensor,
              **kwargs):
    ...
  
  def forward(self,
              observation: th.Tensor,
              action: th.Tensor,
              **kwargs):
    if not isinstance(action, th.Tensor):
      raise ValueError()    
    return self.predict(observation,
                        action,
                        **kwargs)


class _MLP(Base):

  DEFAULT_DIMS = [256, 256, 256]
  DEFAULT_ACTIVATIONS = ["leaky-relu", 
                         "leaky-relu", 
                         "leaky-relu", 
                         "tanh"]

  def __init__(self,
               observation_space: Union[spaces.Dict, spaces.Box],
               action_space: spaces.Box,
               dims: List[int] = DEFAULT_DIMS,
               activations: List[str] = DEFAULT_ACTIVATIONS):
    super().__init__(observation_space, action_space)

    odim = observation_dim(observation_space)
    adim = action_dim(action_space)
    self._body = (
      MLP(dims=[odim + adim, *dims, 1],
           activations=activations).float().to(thu.device()))

  def predict(self, 
              observation: th.Tensor, 
              action: th.Tensor):
    return self._body(th.cat([observation, action], dim=-1))


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
    def thunk():
      return _MLP(observation_space,
                  action_space,
                  dims=dims, 
                  activations=activations)
    self._body = Fusion([thunk for _ in range(fusion)])

  def predict(self, 
              observation: th.Tensor, 
              action: th.Tensor,
              *, 
              member: int = -1, 
              reduce: str = "mean"):
    if member != -1:
      return self._body[member](observation, action)
    else:
      return self._body(observation, action, reduce=reduce)


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
              observation: th.Tensor, 
              action: th.Tensor, 
              **kwargs):
    input = th.cat([observation, action], dim=-1)
    z = self._common_body(input)
    psi, phi = self._psi(z), self._phi(z)
    return th.linalg.vector_norm(psi - phi, 
                                 dim=-1, keepdim=True)


class DistanceL2(Base):

  def __init__(self,
               observation_space,
               action_space,
               *,
               common_dims: List[int] = [256, 256, 256],
               common_activations: List[str] = ["leaky-relu", "leaky-relu", "tanh"],
               projection_dims: List[int] = [128, 2],
               projection_activations: List[int] = ["tanh", "tanh"],
               output_activation: str = "identity"):
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

    from rldev.utils.registry import get
    self._output_activation = get(output_activation)()
  
  def predict(self, observation, action):
    z = self._common_body(th.cat([observation, action], dim=-1))
    psi, phi = self._psi(z), self._phi(z)
    return self._output_activation(
      th.linalg.vector_norm(psi - phi, 
                            dim=-1, keepdim=True))


class FusionDistanceL2(Base):

  def __init__(self, 
               observation_space: Union[spaces.Dict, spaces.Box], 
               action_space: spaces.Box,
               *,
               fusion: int = 3,
               common_dims: List[int] = [256, 256, 256],
               common_activations: List[str] = ["leaky-relu", "leaky-relu", "tanh"],
               projection_dims: List[int] = [128, 2],
               projection_activations: List[int] = ["tanh", "tanh"],
               output_activation: str = "identity"):
    super().__init__(observation_space, action_space)
    def thunk():
      return DistanceL2(observation_space,
                        action_space,
                        common_dims=common_dims,
                        common_activations=common_activations,
                        projection_dims=projection_dims,
                        projection_activations=projection_activations,
                        output_activation=output_activation)
    self._body = Fusion([thunk for _ in range(fusion)])

  def predict(self, 
              observation: th.Tensor, 
              action: th.Tensor,
              *, 
              member: int = -1, 
              reduce: str = "mean"):
    if member != -1:
      return self._body[member](observation, action)
    else:
      return self._body(observation, action, reduce=reduce)
