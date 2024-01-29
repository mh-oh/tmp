
import numpy as np
import pickle
import torch as th

from abc import *
from overrides import overrides
from pathlib import Path
from torch import nn

from rldev.utils import torch as ptu
from rldev.utils.nn import *
from rldev.utils.typing import List, Obs, Union


class ActorCritic:

  def __init__(self, 
               pi: nn.Module,
               pi_lr: float,
               pi_weight_decay: float,
               qfuncs: Union[nn.Module, List[nn.Module]],
               qf_lr: float,
               qf_weight_decay: float):

    self._pi = pi
    self._pi_target = frozen_copy(pi)
    self._pi_parameters = list(pi.parameters())

    if not isinstance(qfuncs, list):
      qfuncs = [qfuncs]

    self._qf = []
    self._qf_target = []
    self._qf_parameters = []
    for qf in qfuncs:
      self._qf.append(qf)
      self._qf_target.append(frozen_copy(qf))
      self._qf_parameters.extend(qf.parameters())

    self._pi_optimizer = (
      th.optim.Adam(self._pi_parameters,
                    lr=pi_lr, weight_decay=pi_weight_decay))
    self._qf_optimizer = (
      th.optim.Adam(self._qf_parameters,
                    lr=qf_lr, weight_decay=qf_weight_decay))

  @property
  def targets_and_models(self):
    yield self._pi_target, self._pi
    for qf_target, qf in zip(self._qf_target, self._qf):
      yield qf_target, qf

  @abstractmethod
  def __call__(self, observation, **kwargs):
    ...


class Policy(ActorCritic):
  u"""DDPG and TD3 variants."""

  def __init__(self,
               max_action: float,
               pi: Actor,
               pi_lr: float,
               pi_weight_decay: float,
               qf: Critic,
               qf_lr: float,
               qf_weight_decay: float):
    
    super().__init__(pi,
                     pi_lr,
                     pi_weight_decay,
                     qf,
                     qf_lr,
                     qf_weight_decay)
    self._max_action = max_action

  @property
  def pi(self):
    return self._pi

  @property
  def pi_target(self):
    return self._pi_target

  @property
  def qf(self):
    return self._qf[0]
  
  @property
  def qf_target(self):
    return self._qf_target[0]

  @overrides
  def __call__(self, observation: Obs[np.ndarray]):
    return ptu.numpy(self.pi(ptu.torch(observation)))
  
  @abstractmethod
  def optimize_batch(self,
                     observation,
                     action,
                     reward,
                     next_observation,
                     done):
    ...


