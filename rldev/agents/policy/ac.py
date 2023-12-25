
import numpy as np
import pickle
import torch as th

from abc import *
from overrides import overrides
from pathlib import Path
from torch import nn
from typing import *

from rldev.agents.core import Node, OffPolicyAgent
from rldev.utils import torch as ptu
from rldev.utils.env import flatten_state
from rldev.utils.nn import *


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


class Policy(Node, ActorCritic):
  u"""DDPG and TD3 variants."""

  def __init__(self,
               agent: OffPolicyAgent, 
               max_action: float,
               pi: Actor,
               pi_lr: float,
               pi_weight_decay: float,
               qf: Critic,
               qf_lr: float,
               qf_weight_decay: float):
    
    Node.__init__(self, agent)
    ActorCritic.__init__(self,
                         pi,
                         pi_lr,
                         pi_weight_decay,
                         qf,
                         qf_lr,
                         qf_weight_decay)
    self._max_action = max_action

  @overrides
  @abstractmethod
  def save(self, dir: Path):
    dir.mkdir(parents=True, exist_ok=True)

    with open(dir / "_pi_parameters.pkl", "wb") as fout:
      pickle.dump(self._pi_parameters, fout)
    with open(dir / "_qf_parameters.pkl", "wb") as fout:
      pickle.dump(self._qf_parameters, fout)
    
    th.save(self._pi_optimizer, dir / "_pi_optimizer.pt")
    th.save(self._pi, dir / "_pi.pt")
    th.save(self._pi_target, dir / "_pi_target.pt")

    th.save(self._qf_optimizer, dir / "_qf_optimizer.pt")
    for i, qf in enumerate(self._qf):
      th.save(qf, dir / f"_qf_{i}.pt")
    for i, qf_target in enumerate(self._qf_target):
      th.save(qf_target, dir / f"_qf_target_{i}.pt")

  @overrides
  @abstractmethod
  def load(self, dir: Path):

    with open(dir / "_pi_parameters.pkl", "rb") as fin:
      self._pi_parameters = pickle.load(fin)
    with open(dir / "_qf_parameters.pkl", "rb") as fin:
      self._qf_parameters = pickle.load(fin)
    
    def load(file):
      return lambda x: x.load_state_dict(dir / file)

    load("_pi_optimizer.pt")(self._pi_optimizer)
    load("_pi.pt")(self._pi)
    load("_pi_target.pt")(self._pi_target)

    load("_qf_optimizer.pt")(self._qf_optimizer)
    for i, qf in enumerate(self._qf):
      load(f"_qf_{i}.pt")(self._qf[i])
    for i, qf_target in enumerate(self._qf_target):
      load(f"_qf_target_{i}.pt")(self._qf_target[i])

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
  def __call__(self, 
               observation: Union[th.Tensor, Dict[str, th.Tensor]],
               greedy: bool = False):

    action_scale = self._max_action
    agent = self.agent
    config = agent.config

    # initial exploration and intrinsic curiosity
    res = None
    if agent.training:
      if config.get('initial_explore') and len(agent.buffer) < config.initial_explore:
        res = np.array([agent._env.action_space.sample() for _ in range(agent._env.num_envs)])
      elif hasattr(agent, 'ag_curiosity'):
        observation = agent.ag_curiosity.relabel_state(observation)
        
    observation = flatten_state(
      observation, config.modalities + config.goal_modalities)
    if agent._observation_normalizer is not None:
      observation = agent._observation_normalizer(observation, update=agent.training)

    if res is not None:
      return res

    observation = ptu.torch(observation)

    if config.get('use_actor_target'):
      action = ptu.numpy(self.pi_target(observation))
    else:
      action = ptu.numpy(self.pi(observation))

    if agent.training and not greedy:
      action = agent._action_noise(action)
      if config.get('eexplore'):
        eexplore = config.eexplore
        if hasattr(agent, 'ag_curiosity'):
          eexplore = agent.ag_curiosity.go_explore * config.go_eexplore + eexplore
        mask = (np.random.random((action.shape[0], 1)) < eexplore).astype(np.float32)
        randoms = np.random.random(action.shape) * (2 * action_scale) - action_scale
        action = mask * randoms + (1 - mask) * action

    return np.clip(action, -action_scale, action_scale)
  
  @abstractmethod
  def optimize_batch(self,
                     observations,
                     actions,
                     rewards,
                     next_observations,
                     gammas):
    ...


class StochasticPolicy(Node, ActorCritic):
  u"""SAC variants."""

  def __init__(self,
               agent: OffPolicyAgent, 
               max_action: float,
               pi: StochasticActor,
               pi_lr: float,
               pi_weight_decay: float,
               qf: Union[Critic, List[Critic]],
               qf_lr: float,
               qf_weight_decay: float):
    
    Node.__init__(self, agent)
    ActorCritic.__init__(self,
                         pi,
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
    return self._qf
  
  @property
  def qf_target(self):
    return self._qf_target

  @overrides
  def __call__(self, observation, greedy=False, **kwargs):

    action_scale = self._max_action
    agent = self.agent
    config = agent.config

    # initial exploration and intrinsic curiosity
    res = None
    if agent.training:
      if config.get('initial_explore') and len(agent.buffer) < config.initial_explore:
          res = np.array([agent._env.action_space.sample() for _ in range(agent._env.num_envs)])
      elif hasattr(agent, 'ag_curiosity'):
        observation = agent.ag_curiosity.relabel_state(observation)
      
    observation = flatten_state(
      observation, config.modalities + config.goal_modalities)  # flatten goal environments
    if agent._observation_normalizer is not None:
      observation = agent._observation_normalizer(observation, update=agent.training)
    
    if res is not None:
      return res

    observation = ptu.torch(observation)

    if config.get('use_actor_target'):
      action, _ = self.pi_target(observation)
    else:
      action, _ = self.pi(observation)
    action = ptu.numpy(action)

    if agent.training and not greedy and config.get('eexplore'):
      eexplore = config.eexplore
      if hasattr(agent, 'ag_curiosity'):
        eexplore = agent.ag_curiosity.go_explore * config.go_eexplore + eexplore
      mask = (np.random.random((action.shape[0], 1)) < eexplore).astype(np.float32)
      randoms = np.random.random(action.shape) * (2 * action_scale) - action_scale
      action = mask * randoms + (1 - mask) * action
    
    return np.clip(action, -action_scale, action_scale)

  @abstractmethod
  def optimize_batch(self,
                     observations,
                     actions,
                     rewards,
                     next_observations,
                     gammas):
    ...