
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from abc import *
from overrides import overrides
from pathlib import Path

from rldev.agents.common.normalizer import *
from rldev.agents.core import OffPolicyAgent
from rldev.agents.policy.ac import Policy
from rldev.utils.nn import soft_update


class DDPGPolicy(Policy):

  def __init__(self, 
               agent: OffPolicyAgent, 
               max_action: float, 
               pi: nn.Module, 
               pi_lr: float, 
               pi_weight_decay: float, 
               qf: nn.Module, 
               qf_lr: float, 
               qf_weight_decay: float):
    super().__init__(agent, 
                     max_action, 
                     pi, 
                     pi_lr, 
                     pi_weight_decay, 
                     qf, 
                     qf_lr, 
                     qf_weight_decay)

  @overrides
  def save(self, dir: Path): super().save(dir)

  @overrides
  def load(self, dir: Path): super().load(dir)

  @overrides
  def optimize_batch(self,
                     observations,
                     actions,
                     rewards,
                     next_observations,
                     gammas):

    action_scale = self._max_action

    agent = self.agent
    config = agent.config
    logger = agent.logger

    critic = self.qf
    critic_target = self.qf_target
    critic_parameters = self._qf_parameters
    critic_optimizer = self._qf_optimizer
    
    actor = self.pi
    actor_target = self.pi_target
    actor_parameters = self._pi_parameters
    actor_optimizer = self._pi_optimizer

    with th.no_grad():
      q_next = critic_target(next_observations, actor_target(next_observations))
      target = (rewards + gammas * q_next)
      target = th.clamp(target, *config.clip_target_range)

    # if config.opt_steps % 1000 == 0:
    #   logger.add_histogram('Optimize/Target_q', target)
    
    q = critic(observations, actions)
    critic_loss = F.mse_loss(q, target)

    critic_optimizer.zero_grad()
    critic_loss.backward()
      
    # Grad clipping
    if config.grad_norm_clipping > 0.:	
      for p in critic_parameters:
        clip_coef = config.grad_norm_clipping / (1e-6 + th.norm(p.grad.detach()))
        if clip_coef < 1:
          p.grad.detach().mul_(clip_coef)
    if config.grad_value_clipping > 0.:
      th.nn.utils.clip_grad_value_(critic_parameters, self.config.grad_value_clipping)

    critic_optimizer.step()

    for p in critic_parameters:
      p.requires_grad = False

    a = actor(observations)
    if config.get('policy_opt_noise'):
      noise = th.randn_like(a) * (config.policy_opt_noise * action_scale)
      a = (a + noise).clamp(-action_scale, action_scale)
      
    actor_loss = -critic(observations, a)[:, -1].mean()
    if config.action_l2_regularization:
      actor_loss += config.action_l2_regularization * F.mse_loss(a / action_scale, th.zeros_like(a))

    actor_optimizer.zero_grad()
    actor_loss.backward()
      
    # Grad clipping
    if config.grad_norm_clipping > 0.:	
      for p in actor_parameters:
        clip_coef = config.grad_norm_clipping / (1e-6 + th.norm(p.grad.detach()))
        if clip_coef < 1:
          p.grad.detach().mul_(clip_coef)
    if config.grad_value_clipping > 0.:
      th.nn.utils.clip_grad_value_(actor_parameters, config.grad_value_clipping)
      
    actor_optimizer.step()

    for p in critic_parameters:
      p.requires_grad = True


class DDPG(OffPolicyAgent):

  def __init__(
      self,
      config,
      env,
      test_env,
      policy: DDPGPolicy,
      buffer,
      observation_normalizer=None,
      action_noise=None):
    super().__init__(config, env, test_env, policy, buffer)

    def wrap(fn):
      return (lambda x: None) if fn is None else fn

    self._observation_normalizer = wrap(observation_normalizer)(self)
    self._action_noise = wrap(action_noise)(self)
  
  @overrides
  def save(self): super().save()

  @overrides
  def load(self, dir: Path): super().load(dir)

  @overrides
  def process_episodic_records(self, done):
    return super().process_episodic_records(done)

  @overrides
  def process_experience(self, experience):
    self._buffer._process_experience(experience)

  @overrides
  def optimize(self):

    config = self.config
    buffer = self.buffer

    if len(buffer) > config.warm_up:
      self._policy.optimize_batch(*buffer.sample(config.batch_size))
      if self.opt_steps % config.target_network_update_freq == 0:
        for target, model in self._policy.targets_and_models:
          soft_update(target, model, config.target_network_update_frac)