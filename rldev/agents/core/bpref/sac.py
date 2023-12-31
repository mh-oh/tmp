
import math
import numpy as np
import torch
import torch.nn.functional as F

from overrides import overrides
from pathlib import Path
from torch import distributions as pyd
from torch import nn

from rldev.agents.core import Node
from rldev.agents.core.bpref import utils
from rldev.utils import torch as thu


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
               observation_space, 
               action_space, 
               hidden_dim, 
               hidden_depth,
               log_std_bounds):
    super().__init__()

    obs_dim = observation_space.shape[0]
    action_dim = action_space.shape[0]
    self.log_std_bounds = log_std_bounds
    self.trunk = utils.mlp(obs_dim, hidden_dim, 2 * action_dim, hidden_depth)

    self.outputs = dict()
    self.apply(utils.weight_init)

  def forward(self, obs):
    mu, log_std = self.trunk(obs).chunk(2, dim=-1)

    # constrain log_std inside [log_std_min, log_std_max]
    log_std = torch.tanh(log_std)
    log_std_min, log_std_max = self.log_std_bounds
    log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)

    std = log_std.exp()

    self.outputs['mu'] = mu
    self.outputs['std'] = std

    dist = SquashedNormal(mu, std)
    return dist


############################################################
############################################################
############################################################
############################################################
############################################################
                
import numpy as np
import torch
import torch.nn.functional as F
from rldev.agents.core.bpref import utils

from torch import nn

class DoubleQCritic(nn.Module):
  """Critic network, employes double Q-learning."""

  def __init__(self, 
               observation_space, action_space, hidden_dim, hidden_depth):
    super().__init__()

    obs_dim = observation_space.shape[0]
    action_dim = action_space.shape[0]
    self.Q1 = utils.mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)
    self.Q2 = utils.mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)

    self.outputs = dict()
    self.apply(utils.weight_init)

  def forward(self, obs, action):
    assert obs.size(0) == action.size(0)

    obs_action = torch.cat([obs, action], dim=-1)
    q1 = self.Q1(obs_action)
    q2 = self.Q2(obs_action)

    self.outputs['q1'] = q1
    self.outputs['q2'] = q2

    return q1, q2

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
  """SAC algorithm."""
  def __init__(self, 
               agent, 
               observation_space,
               action_space, 
               discount, 
               init_temperature, 
               alpha_lr, 
               alpha_betas,
               actor_lr, 
               actor_betas, 
               actor_update_frequency, 
               critic_lr,
               critic_betas, 
               critic_tau, 
               critic_target_update_frequency,
               batch_size, 
               learnable_temperature,
               qf_hidden_dim,
               qf_hidden_depth,
               pi_hidden_dim,
               pi_hidden_depth,
               pi_log_std_bounds,
               normalize_state_entropy=True):
    super().__init__(agent)

    self.action_range = [float(action_space.low.min()), float(action_space.high.max())]
    self.device = thu.device()
    self.discount = discount
    self.critic_tau = critic_tau
    self.actor_update_frequency = actor_update_frequency
    self.critic_target_update_frequency = critic_target_update_frequency
    self.batch_size = batch_size
    self.learnable_temperature = learnable_temperature

    self.critic_lr = critic_lr
    self.critic_betas = critic_betas
    self.s_ent_stats = utils.TorchRunningMeanStd(shape=[1], device=thu.device())
    self.normalize_state_entropy = normalize_state_entropy
    self.init_temperature = init_temperature
    self.alpha_lr = alpha_lr
    self.alpha_betas = alpha_betas
    self.actor_betas = actor_betas
    self.alpha_lr = alpha_lr

    self.qf_kwargs = {"observation_space": observation_space,
                      "action_space": action_space,
                      "hidden_dim": qf_hidden_dim,
                      "hidden_depth": qf_hidden_depth}
    self.critic = DoubleQCritic(**self.qf_kwargs).to(self.device)
    self.critic_target = DoubleQCritic(**self.qf_kwargs).to(self.device)
    self.critic_target.load_state_dict(self.critic.state_dict())

    self.pi_kwargs = {"observation_space": observation_space,
                      "action_space": action_space,
                      "hidden_dim": pi_hidden_dim,
                      "hidden_depth": pi_hidden_depth,
                      "log_std_bounds": pi_log_std_bounds}
    self.actor = DiagGaussianActor(**self.pi_kwargs).to(self.device)
    self.log_alpha = torch.tensor(np.log(init_temperature)).to(self.device)
    self.log_alpha.requires_grad = True
    
    # set target entropy to -|A|
    assert len(action_space.shape) == 1
    self.target_entropy = -action_space.shape[0]

    # optimizers
    self.actor_optimizer = torch.optim.Adam(
        self.actor.parameters(),
        lr=actor_lr,
        betas=actor_betas)
    self.critic_optimizer = torch.optim.Adam(
        self.critic.parameters(),
        lr=critic_lr,
        betas=critic_betas)
    self.log_alpha_optimizer = torch.optim.Adam(
        [self.log_alpha],
        lr=alpha_lr,
        betas=alpha_betas)
    
    # change mode
    self.train()
    self.critic_target.train()
  
  def reset_critic(self):
    self.critic = DoubleQCritic(**self.qf_kwargs).to(self.device)
    self.critic_target = DoubleQCritic(**self.qf_kwargs).to(self.device)
    self.critic_target.load_state_dict(self.critic.state_dict())
    self.critic_optimizer = torch.optim.Adam(
        self.critic.parameters(),
        lr=self.critic_lr,
        betas=self.critic_betas)
  
  def reset_actor(self):
    # reset log_alpha
    self.log_alpha = torch.tensor(np.log(self.init_temperature)).to(self.device)
    self.log_alpha.requires_grad = True
    self.log_alpha_optimizer = torch.optim.Adam(
        [self.log_alpha],
        lr=self.alpha_lr,
        betas=self.alpha_betas)
    
    # reset actor
    self.actor = DiagGaussianActor(**self.pi_kwargs).to(self.device)
    self.actor_optimizer = torch.optim.Adam(
        self.actor.parameters(),
        lr=self.actor_lr,
        betas=self.actor_betas)
      
  def train(self, training=True):
    self.training = training
    self.actor.train(training)
    self.critic.train(training)

  @property
  def alpha(self):
      return self.log_alpha.exp()

  def __call__(self, obs, sample=False):
      return self.act(obs, sample)

  def act(self, obs, sample=False):
    __obs = obs
    obs = torch.FloatTensor(obs).to(self.device)
    dist = self.actor(obs)
    action = dist.sample() if sample else dist.mean
    action = action.clamp(*self.action_range)
    if not (action.ndim == 2):
      raise AssertionError(f"{action.shape}, {__obs.shape}")
    return utils.to_np(action)

  def update_critic(self, obs, action, reward, next_obs, 
                    not_done, logger, step, print_flag=True):
      
    dist = self.actor(next_obs)
    next_action = dist.rsample()
    log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
    target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
    target_V = torch.min(target_Q1,
                          target_Q2) - self.alpha.detach() * log_prob
    target_Q = reward + (not_done * self.discount * target_V)
    target_Q = target_Q.detach()

    # get current Q estimates
    current_Q1, current_Q2 = self.critic(obs, action)
    critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
        current_Q2, target_Q)
    
    # if print_flag:
    #     logger.log('train_critic/loss', critic_loss, step)

    # Optimize the critic
    self.critic_optimizer.zero_grad()
    critic_loss.backward()
    self.critic_optimizer.step()
      
  def update_critic_state_ent(
      self, obs, full_obs, action, next_obs, not_done, logger,
      step, K=5, print_flag=True):
      
    dist = self.actor(next_obs)
    next_action = dist.rsample()
    log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
    target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
    target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_prob
    
    # compute state entropy
    state_entropy = compute_state_entropy(obs, full_obs, k=K)
    # if print_flag:
    #     logger.log("train_critic/entropy", state_entropy.mean(), step)
    #     logger.log("train_critic/entropy_max", state_entropy.max(), step)
    #     logger.log("train_critic/entropy_min", state_entropy.min(), step)
    
    self.s_ent_stats.update(state_entropy)
    norm_state_entropy = state_entropy / self.s_ent_stats.std
    
    # if print_flag:
    #     logger.log("train_critic/norm_entropy", norm_state_entropy.mean(), step)
    #     logger.log("train_critic/norm_entropy_max", norm_state_entropy.max(), step)
    #     logger.log("train_critic/norm_entropy_min", norm_state_entropy.min(), step)
    
    if self.normalize_state_entropy:
      state_entropy = norm_state_entropy
    
    target_Q = state_entropy + (not_done * self.discount * target_V)
    target_Q = target_Q.detach()

    # get current Q estimates
    current_Q1, current_Q2 = self.critic(obs, action)
    critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
        current_Q2, target_Q)
    
    # if print_flag:
    #     logger.log('train_critic/loss', critic_loss, step)

    # Optimize the critic
    self.critic_optimizer.zero_grad()
    critic_loss.backward()
    self.critic_optimizer.step()
  
  @overrides
  def save(self, dir: Path): ...

  @overrides
  def load(self, dir: Path): ...
  
  def update_actor_and_alpha(self, obs, logger, step, print_flag=False):
    dist = self.actor(obs)
    action = dist.rsample()
    log_prob = dist.log_prob(action).sum(-1, keepdim=True)
    actor_Q1, actor_Q2 = self.critic(obs, action)

    actor_Q = torch.min(actor_Q1, actor_Q2)
    actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()
    # if print_flag:
    #     logger.log('train_actor/loss', actor_loss, step)
    #     logger.log('train_actor/target_entropy', self.target_entropy, step)
    #     logger.log('train_actor/entropy', -log_prob.mean(), step)

    # optimize the actor
    self.actor_optimizer.zero_grad()
    actor_loss.backward()
    self.actor_optimizer.step()

    if self.learnable_temperature:
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

    if step % self.critic_target_update_frequency == 0:
      utils.soft_update_params(self.critic, self.critic_target,
                                self.critic_tau)
          
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

    if step % self.critic_target_update_frequency == 0:
      utils.soft_update_params(self.critic, self.critic_target,
                                self.critic_tau)