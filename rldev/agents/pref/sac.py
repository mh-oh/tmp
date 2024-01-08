
import math
import numpy as np
import torch
import torch as th

from gymnasium import spaces
from overrides import overrides
from pathlib import Path
from typing import *

from torch import distributions as pyd
from torch import nn
from torch.nn import functional as F

from rldev.agents.core import Node, Agent
from rldev.agents.pref import utils
from rldev.utils import gym_types
from rldev.utils import torch as thu
from rldev.utils.env import flatten_space
from rldev.utils.nn import _MLP as MLP


class TanhTransform(pyd.transforms.Transform):

  domain = pyd.constraints.real
  codomain = pyd.constraints.interval(-1.0, 1.0)
  bijective = True
  sign = +1

  def __init__(self, cache_size=1):
    super().__init__(cache_size=cache_size)

  @staticmethod
  def atanh(x):
    return 0.5 * (x.log1p() - (-x).log1p())

  def __eq__(self, other):
    return isinstance(other, TanhTransform)

  def _call(self, x):
    return x.tanh()

  def _inverse(self, y):
    # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
    # one should use `cache_size=1` instead
    return self.atanh(y)

  def log_abs_det_jacobian(self, x, y):
    # We use a formula that is more numerically stable, see details in the following link
    # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
    return 2. * (math.log(2.) - x - F.softplus(-2. * x))


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    
  def __init__(self, loc, scale):
    self.loc = loc
    self.scale = scale

    self.base_dist = pyd.Normal(loc, scale)
    transforms = [TanhTransform()]
    super().__init__(self.base_dist, transforms)

  @property
  def mean(self):
    mu = self.loc
    for tr in self.transforms:
      mu = tr(mu)
    return mu


class DiagGaussianActor(nn.Module):
  """torch.distributions implementation of an diagonal Gaussian policy."""
  def __init__(self, 
               observation_space: spaces.Box, 
               action_space: spaces.Box, 
               *,
               dims: List[int],
               lr: float,
               betas: List[float],
               decay: float,
               log_std_bounds: List[float]):
    super().__init__()

    odim, = observation_space.shape
    adim, = action_space.shape
    self.log_std_bounds = log_std_bounds

    activations = ["relu"] * len(dims) + ["identity"]
    self._pi = MLP([odim, *dims, 2 * adim], activations)

    self.apply(utils.weight_init)

    self._optimizer = (
      th.optim.Adam(self._pi.parameters(),
                    lr=lr, betas=betas, weight_decay=decay))

  @property
  def optimizer(self):
    return self._optimizer

  def save(self, dir: Path):
    dir.mkdir(parents=True, exist_ok=True)
    th.save(self._optimizer.state_dict(), dir / "_optimizer.pt")
    th.save(self._pi.state_dict(), dir / "_pi.pt")

  def load(self, dir: Path):
    ...

  def forward(self, obs):
    mu, log_std = self._pi(obs).chunk(2, dim=-1)

    # constrain log_std inside [log_std_min, log_std_max]
    log_std = th.tanh(log_std)
    log_std_min, log_std_max = self.log_std_bounds
    log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)

    dist = SquashedNormal(mu, log_std.exp())
    return dist


class QFunction(nn.Module):

  class Funcs(nn.Module):

    def __init__(self, dims, n):
      super().__init__()
      activations = ["relu"] * (len(dims) - 2) + ["identity"]
      self._body = nn.ModuleList([
          MLP(dims, activations=activations) 
            for _ in range(n)])
      self.apply(utils.weight_init)
    
    def forward(self, observation, action):
      return [f(torch.cat([observation, action], dim=-1)) 
              for f in self._body]

  def __init__(self, 
               observation_space: spaces.Box, 
               action_space: spaces.Box, 
               *,
               dims: List[int],
               lr: float,
               betas: List[float],
               decay: float,
               n_qfuncs: int = 2):
    super().__init__()

    odim, = observation_space.shape
    adim, = action_space.shape

    def qfuncs():
      return QFunction.Funcs([odim + adim, *dims, 1], n_qfuncs)    

    self._qf = qfuncs()
    self._qf_target = qfuncs()
    self._qf_target.load_state_dict(self._qf.state_dict())

    self._optimizer = (
      th.optim.Adam(self._qf.parameters(),
                    lr=lr, betas=betas, weight_decay=decay))

  @property
  def target(self):
    return self._qf_target

  @property
  def optimizer(self):
    return self._optimizer

  def save(self, dir: Path):
    dir.mkdir(parents=True, exist_ok=True)
    th.save(self._optimizer.state_dict(), dir / "_optimizer.pt")
    th.save(self._qf.state_dict(), dir / "_qf.pt")
    th.save(self._qf_target.state_dict(), dir / "_qf_target.pt")

  def load(self, dir: Path):
    ...

  def forward(self, observation, action):
    return self._qf(observation, action)

############################################################
############################################################
                
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def compute_state_entropy(obs, full_obs, k):
  batch_size = 500
  with torch.no_grad():
    dists = []
    for idx in range(len(full_obs) // batch_size + 1):
      start = idx * batch_size
      end = (idx + 1) * batch_size
      dist = torch.norm(
          obs[:, None, :] - full_obs[None, start:end, :], dim=-1, p=2
      )
      dists.append(dist)

    dists = torch.cat(dists, dim=1)
    knn_dists = torch.kthvalue(dists, k=k + 1, dim=1).values
    state_entropy = knn_dists
  return state_entropy.unsqueeze(1)

class SACPolicy(Node):

  def __init__(self, 
               agent: Agent, 
               observation_space: Union[spaces.Box, spaces.Dict],
               action_space: spaces.Box, 
               batch_size: int, 
               discount: float, 
               tau: float,
               qf_cls: type,
               qf_kwargs: Dict[str, Any],
               critic_target_update_frequency: int,
               pi_cls: type,
               pi_kwargs: Dict[str, Any],
               actor_update_frequency: int, 
               learnable_alpha: Tuple[bool, Dict[str, Any]],
               normalize_state_entropy: bool = True):
    super().__init__(agent)

    if isinstance(observation_space, gym_types.Dict):
      observation_space = flatten_space(observation_space)

    self._observation_space = observation_space
    self._action_space = action_space

    self.action_range = [float(action_space.low.min()), float(action_space.high.max())]
    self.device = thu.device()
    self.discount = discount
    self._tau = tau
    self.actor_update_frequency = actor_update_frequency
    self._qf_target_update_frequency = critic_target_update_frequency
    self.batch_size = batch_size

    self.s_ent_stats = utils.TorchRunningMeanStd(shape=[1], device=thu.device())
    self.normalize_state_entropy = normalize_state_entropy
    
    self._learnable_alpha, kwargs = learnable_alpha
    self.init_temperature = kwargs["init"]
    self.alpha_lr = kwargs["lr"]
    self.alpha_betas = kwargs["betas"]

    self._qf_cls = qf_cls
    self._qf_kwargs = qf_kwargs
    self._qf = qf_cls(observation_space,
                      action_space, **qf_kwargs).to(self.device)

    self._pi_cls = pi_cls
    self._pi_kwargs = pi_kwargs
    self._pi = self._pi_cls(
      observation_space, action_space, **pi_kwargs).to(self.device)

    self.log_alpha = th.tensor(np.log(self.init_temperature)).to(self.device)
    self.log_alpha.requires_grad = True
    
    # set target entropy to -|A|
    assert len(action_space.shape) == 1
    self.target_entropy = -action_space.shape[0]

    # optimizers
    self.log_alpha_optimizer = (
      th.optim.Adam([self.log_alpha],
                    lr=self.alpha_lr, betas=self.alpha_betas))
    
    # change mode
    self.train()
  
  def reset_critic(self):
    self._qf = self._qf_cls(
      self._observation_space, self._action_space, **self._qf_kwargs).to(self.device)
  
  def reset_actor(self):
    # reset log_alpha
    self.log_alpha = torch.tensor(np.log(self.init_temperature)).to(self.device)
    self.log_alpha.requires_grad = True
    self.log_alpha_optimizer = torch.optim.Adam(
        [self.log_alpha],
        lr=self.alpha_lr,
        betas=self.alpha_betas)
    
    # reset actor
    self._pi = self._pi_cls(
      self._observation_space, self._action_space, **self._pi_kwargs).to(self.device)
      
  def train(self, training=True):
    self.training = training
    self._pi.train(training)
    self._qf.train(training)

  @property
  def alpha(self):
      return self.log_alpha.exp()

  def __call__(self, obs, sample=False):
      return self.act(obs, sample)

  def act(self, obs, sample=False):
    __obs = obs
    obs = torch.FloatTensor(obs).to(self.device)
    dist = self._pi(obs)
    action = dist.sample() if sample else dist.mean
    action = action.clamp(*self.action_range)
    if not (action.ndim == 2):
      raise AssertionError(f"{action.shape}, {__obs.shape}")
    return utils.to_np(action)

  def update_critic(self, obs, action, reward, next_obs, 
                    not_done, logger, step, print_flag=True):
      
    dist = self._pi(next_obs)
    next_action = dist.rsample()
    log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
    target_Q1, target_Q2 = self._qf.target(next_obs, next_action)
    target_V = torch.min(target_Q1,
                          target_Q2) - self.alpha.detach() * log_prob
    target_Q = reward + (not_done * self.discount * target_V)
    target_Q = target_Q.detach()

    # get current Q estimates
    current_Q1, current_Q2 = self._qf(obs, action)
    critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
        current_Q2, target_Q)
    
    # if print_flag:
    #     logger.log('train_critic/loss', critic_loss, step)

    # Optimize the critic
    self._qf.optimizer.zero_grad()
    critic_loss.backward()
    self._qf.optimizer.step()
      
  def update_critic_state_ent(
      self, obs, full_obs, action, next_obs, not_done, logger,
      step, K=5, print_flag=True):
      
    dist = self._pi(next_obs)
    next_action = dist.rsample()
    log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
    target_Q1, target_Q2 = self._qf.target(next_obs, next_action)
    target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_prob
    
    # compute state entropy
    state_entropy = compute_state_entropy(obs, full_obs, k=K)
    
    self.s_ent_stats.update(state_entropy)
    norm_state_entropy = state_entropy / self.s_ent_stats.std
    
    if self.normalize_state_entropy:
      state_entropy = norm_state_entropy
    
    target_Q = state_entropy + (not_done * self.discount * target_V)
    target_Q = target_Q.detach()

    # get current Q estimates
    current_Q1, current_Q2 = self._qf(obs, action)
    critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
        current_Q2, target_Q)
    
    # if print_flag:
    #     logger.log('train_critic/loss', critic_loss, step)

    # Optimize the critic
    self._qf.optimizer.zero_grad()
    critic_loss.backward()
    self._qf.optimizer.step()
  
  @overrides
  def save(self, dir: Path):
    dir.mkdir(parents=True, exist_ok=True)
    self._pi.save(dir / "_pi")
    self._qf.save(dir / "_qf")

  @overrides
  def load(self, dir: Path): ...
  
  def update_actor_and_alpha(self, obs, logger, step, print_flag=False):
    dist = self._pi(obs)
    action = dist.rsample()
    log_prob = dist.log_prob(action).sum(-1, keepdim=True)
    actor_Q1, actor_Q2 = self._qf(obs, action)

    actor_Q = torch.min(actor_Q1, actor_Q2)
    actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()
    # if print_flag:
    #     logger.log('train_actor/loss', actor_loss, step)
    #     logger.log('train_actor/target_entropy', self.target_entropy, step)
    #     logger.log('train_actor/entropy', -log_prob.mean(), step)

    # optimize the actor
    self._pi.optimizer.zero_grad()
    actor_loss.backward()
    self._pi.optimizer.step()

    if self._learnable_alpha:
      self.log_alpha_optimizer.zero_grad()
      alpha_loss = (self.alpha *
                    (-log_prob - self.target_entropy).detach()).mean()
      # if print_flag:
      #     logger.log('train_alpha/loss', alpha_loss, step)
      #     logger.log('train_alpha/value', self.alpha, step)
      alpha_loss.backward()
      self.log_alpha_optimizer.step()
          
  def update(self, replay_buffer, logger, step, gradient_update=1):
    for index in range(gradient_update):
      obs, action, reward, next_obs, not_done, not_done_no_max = replay_buffer.sample(
          self.batch_size)

      print_flag = False
      if index == gradient_update -1:
        # logger.log('train/batch_reward', reward.mean(), step)
        print_flag = True
          
      self.update_critic(obs, action, reward, next_obs, not_done_no_max,
                          logger, step, print_flag)

      if step % self.actor_update_frequency == 0:
        self.update_actor_and_alpha(obs, logger, step, print_flag)

    if step % self._qf_target_update_frequency == 0:
      utils.soft_update_params(self._qf, self._qf.target,
                                self._tau)
          
  def update_state_ent(self, replay_buffer, logger, step, gradient_update=1, K=5):
    for index in range(gradient_update):
      obs, full_obs, action, reward, next_obs, not_done, not_done_no_max = replay_buffer.sample_state_ent(
          self.batch_size)

      print_flag = False
      if index == gradient_update -1:
        # logger.log('train/batch_reward', reward.mean(), step)
        print_flag = True
          
      self.update_critic_state_ent(
          obs, full_obs, action, next_obs, not_done_no_max,
          logger, step, K=K, print_flag=print_flag)

      if step % self.actor_update_frequency == 0:
        self.update_actor_and_alpha(obs, logger, step, print_flag)

    if step % self._qf_target_update_frequency == 0:
      utils.soft_update_params(self._qf, self._qf.target,
                                self._tau)