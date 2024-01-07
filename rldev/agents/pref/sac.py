
import math
import numpy as np
import pickle
import torch as th

from overrides import overrides
from pathlib import Path
from torch import distributions as pyd
from torch import nn
from torch.nn import functional as F

from rldev.agents.core import Node
from rldev.agents.pref import utils
from rldev.utils import gym_types
from rldev.utils import torch as thu
from rldev.utils.env import flatten_space
from rldev.utils.nn import _MLP as MLP, frozen_copy


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

  def __init__(self, 
               observation_space, 
               action_space, 
               *,
               dims,
               log_std_bounds,
               lr,
               betas,
               decay):
    super().__init__()

    odim, = observation_space.shape
    adim, = action_space.shape

    self._log_std_bounds = log_std_bounds
    self._pi = MLP([odim, *dims, 2 * adim], activations="relu") 

    self.apply(utils.weight_init)

    self._optimizer = (
      th.optim.Adam(self._pi.parameters(),
                    lr=lr, betas=betas, weight_decay=decay))

  def forward(self, observation):
    mu, log_std = self._pi(observation).chunk(2, dim=-1)

    # constrain log_std inside [log_std_min, log_std_max]
    log_std = th.tanh(log_std)
    log_std_min, log_std_max = self._log_std_bounds
    log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)

    return SquashedNormal(mu, log_std.exp())

  def save(self, dir: Path):

    dir.mkdir(parents=True, exist_ok=True)

    th.save(self._optimizer.state_dict(), dir / "_optimizer.pt")
    th.save(self._pi.state_dict(), dir / "_pi.pt")

  def load(self, dir: Path):
    ...


class QFunction(nn.Module):

  def __init__(self,
               observation_space,
               action_space,
               *,
               dims,
               lr,
               betas,
               decay,
               n_funcs=1):
    super().__init__()

    odim, = observation_space.shape
    adim, = action_space.shape
    self._input_dim = odim + adim

    def qfuncs():
      return nn.ModuleList([
        MLP([odim + adim, *dims, 1], activations="relu") 
          for _ in range(n_funcs)])

    self._qf = qfuncs()
    self._target = qfuncs()
    self._target.load_state_dict(self._qf.state_dict())
    for p in self._target.parameters():
      p.requires_grad = False

    self._optimizer = (
      th.optim.Adam(self._qf.parameters(),
                    lr=lr, betas=betas, weight_decay=decay))

  def parameters(self):
    return self._qf.parameters()

  def forward(self, observation, action):

    input = th.cat([observation, action], dim=-1)
    *_, dim = input.shape
    if (dim != self._input_dim):
      raise ValueError(f"")
    return [qf(input) for qf in self._qf]

  def target(self, observation, action):

    input = th.cat([observation, action], dim=-1)
    *_, dim = input.shape
    if (dim != self._input_dim):
      raise ValueError(f"")
    return [qf(input) for qf in self._target]

  def save(self, dir: Path):

    dir.mkdir(parents=True, exist_ok=True)

    th.save(self._optimizer.state_dict(), dir / "_optimizer.pt")
    th.save(self._qf.state_dict(), dir / "_qf.pt")
    th.save(self._target.state_dict(), dir / "_target.pt")

  def load(self, dir: Path):
    ...


############################################################
############################################################

def compute_state_entropy(obs, full_obs, k):
  batch_size = 500
  with th.no_grad():
    dists = []
    for idx in range(len(full_obs) // batch_size + 1):
      start = idx * batch_size
      end = (idx + 1) * batch_size
      dist = th.norm(
          obs[:, None, :] - full_obs[None, start:end, :], dim=-1, p=2
      )
      dists.append(dist)

    dists = th.cat(dists, dim=1)
    knn_dists = th.kthvalue(dists, k=k + 1, dim=1).values
    state_entropy = knn_dists
  return state_entropy.unsqueeze(1)


class SACPolicy(Node):
  """SAC algorithm."""
  def __init__(self,
               agent,
               observation_space,
               action_space,
               tau,
               discount,
               batch_size,
               learnable_alpha,
               qf_cls,
               qf_kwargs,
               update_qf_target_every_n_steps,
               pi_cls,
               pi_kwargs,
               update_pi_every_n_steps,
               normalize_state_entropy=True):
    super().__init__(agent)

    if isinstance(observation_space, gym_types.Dict):
      observation_space = flatten_space(observation_space)

    self._observation_space = observation_space
    self._action_space = action_space

    self.action_range = [float(action_space.low.min()), float(action_space.high.max())]
    self._tau = tau
    self._discount = discount
    self._batch_size = batch_size

    self._learnable_alpha, kwargs = learnable_alpha
    self._alpha_init = kwargs["init"]
    self._alpha_lr = kwargs["lr"]
    self._alpha_betas = kwargs["betas"]

    self._log_alpha = th.tensor(np.log(self._alpha_init)).to(thu.device())
    self._log_alpha.requires_grad = True
    self._log_alpha_optimizer = (
      th.optim.Adam([self._log_alpha],
                     lr=self._alpha_lr,betas=self._alpha_betas))

    self._qf_cls = qf_cls
    self._qf_kwargs = qf_kwargs
    self._qf = qf_cls(observation_space,
                      action_space,
                      **qf_kwargs).to(thu.device())
    self._update_qf_target_every_n_steps = update_qf_target_every_n_steps

    self._pi_cls = pi_cls
    self._pi_kwargs = pi_kwargs
    self._pi = pi_cls(observation_space,
                      action_space,
                      **pi_kwargs).to(thu.device())
    self._update_pi_every_n_steps = update_pi_every_n_steps

    self.s_ent_stats = utils.TorchRunningMeanStd(shape=[1], device=thu.device())
    self.normalize_state_entropy = normalize_state_entropy

    # set target entropy to -|A|
    assert len(action_space.shape) == 1
    self.target_entropy = -action_space.shape[0]
    
    # change mode
    self.train()
  
  def reset_critic(self):
    self._qf = self._qf_cls(self._observation_space,
                            self._action_space,
                            **self._qf_kwargs).to(thu.device())
  
  def reset_actor(self):
    # reset log_alpha
    self._log_alpha = th.tensor(np.log(self._alpha_init)).to(thu.device())
    self._log_alpha.requires_grad = True
    self._log_alpha_optimizer = th.optim.Adam(
        [self._log_alpha],
        lr=self._alpha_lr,
        betas=self._alpha_betas)
    
    # reset actor
    self._pi = self._pi_cls(self._observation_space,
                            self._action_space,
                            **self._pi_kwargs).to(thu.device())
      
  def train(self, training=True):
    self.training = training
    self._pi.train(training)
    self._qf.train(training)

  @property
  def alpha(self):
      return self._log_alpha.exp()

  def __call__(self, obs, sample=False):
      return self.act(obs, sample)

  def act(self, obs, sample=False):
    __obs = obs
    obs = th.FloatTensor(obs).to(thu.device())
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
    target_V = th.min(target_Q1,
                          target_Q2) - self.alpha.detach() * log_prob
    target_Q = reward + (not_done * self._discount * target_V)
    target_Q = target_Q.detach()

    # get current Q estimates
    current_Q1, current_Q2 = self._qf(obs, action)
    critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
        current_Q2, target_Q)
    
    # Optimize the critic
    self._qf._optimizer.zero_grad()
    critic_loss.backward()
    self._qf._optimizer.step()
      
  def update_critic_state_ent(
      self, obs, full_obs, action, next_obs, not_done, logger,
      step, K=5, print_flag=True):
      
    dist = self._pi(next_obs)
    next_action = dist.rsample()
    log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
    target_Q1, target_Q2 = self._qf.target(next_obs, next_action)
    target_V = th.min(target_Q1, target_Q2) - self.alpha.detach() * log_prob
    
    # compute state entropy
    state_entropy = compute_state_entropy(obs, full_obs, k=K)
    
    self.s_ent_stats.update(state_entropy)
    norm_state_entropy = state_entropy / self.s_ent_stats.std
    
    if self.normalize_state_entropy:
      state_entropy = norm_state_entropy
    
    target_Q = state_entropy + (not_done * self._discount * target_V)
    target_Q = target_Q.detach()

    # get current Q estimates
    current_Q1, current_Q2 = self._qf(obs, action)
    critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
        current_Q2, target_Q)
    
    # Optimize the critic
    self._qf._optimizer.zero_grad()
    critic_loss.backward()
    self._qf._optimizer.step()
  
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

    actor_Q = th.min(actor_Q1, actor_Q2)
    actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

    # optimize the actor
    self._pi._optimizer.zero_grad()
    actor_loss.backward()
    self._pi._optimizer.step()

    if self._learnable_alpha:
      self._log_alpha_optimizer.zero_grad()
      alpha_loss = (self.alpha *
                    (-log_prob - self.target_entropy).detach()).mean()
      alpha_loss.backward()
      self._log_alpha_optimizer.step()
          
  def update(self, replay_buffer, logger, step, gradient_update=1):
    for index in range(gradient_update):
      obs, action, reward, next_obs, not_done, not_done_no_max = replay_buffer.sample(
          self._batch_size)

      print_flag = False
      if index == gradient_update -1:
        print_flag = True
          
      self.update_critic(obs, action, reward, next_obs, not_done_no_max,
                          logger, step, print_flag)

      if step % self._update_pi_every_n_steps == 0:
        self.update_actor_and_alpha(obs, logger, step, print_flag)

    if step % self._update_qf_target_every_n_steps == 0:
      utils.soft_update_params(self._qf._qf, self._qf._target,
                                self._tau)
          
  def update_state_ent(self, replay_buffer, logger, step, gradient_update=1, K=5):
    for index in range(gradient_update):
      obs, full_obs, action, reward, next_obs, not_done, not_done_no_max = replay_buffer.sample_state_ent(
          self._batch_size)

      print_flag = False
      if index == gradient_update -1:
        print_flag = True
          
      self.update_critic_state_ent(
          obs, full_obs, action, next_obs, not_done_no_max,
          logger, step, K=K, print_flag=print_flag)

      if step % self._update_pi_every_n_steps == 0:
        self.update_actor_and_alpha(obs, logger, step, print_flag)

    if step % self._update_qf_target_every_n_steps == 0:
      utils.soft_update_params(self._qf._qf, self._qf._target,
                                self._tau)