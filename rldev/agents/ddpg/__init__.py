
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from abc import *
from overrides import overrides
from pathlib import Path

from rldev.agents import ActionNoise, ObservationNormalizer
from rldev.agents.core import OffPolicyAgent
from rldev.agents.policy.ac import Policy
from rldev.buffers.basic import Buffer
from rldev.feature_extractor import Extractor
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
                     observation,
                     action,
                     reward,
                     next_observation,
                     done):

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
      q_next = critic_target(next_observation, actor_target(next_observation))
      target = (reward + (self._agent._config.gamma * (1. - done)) * q_next)
      target = th.clamp(target, *config.clip_target_range)

    # if config.opt_steps % 1000 == 0:
    #   logger.add_histogram('Optimize/Target_q', target)
    
    q = critic(observation, action)
    critic_loss = F.mse_loss(q, target)

    critic_optimizer.zero_grad()
    critic_loss.backward()
      
    # Grad clipping
    if config.grad_norm_clipping > 0.:	
      raise
      for p in critic_parameters:
        clip_coef = config.grad_norm_clipping / (1e-6 + th.norm(p.grad.detach()))
        if clip_coef < 1:
          p.grad.detach().mul_(clip_coef)
    if config.grad_value_clipping > 0.:
      raise
      th.nn.utils.clip_grad_value_(critic_parameters, self.config.grad_value_clipping)

    critic_optimizer.step()

    for p in critic_parameters:
      p.requires_grad = False

    a = actor(observation)
    if config.get('policy_opt_noise'):
      raise
      noise = th.randn_like(a) * (config.policy_opt_noise * action_scale)
      a = (a + noise).clamp(-action_scale, action_scale)
      
    actor_loss = -critic(observation, a)[:, -1].mean()
    if config.action_l2_regularization:
      actor_loss += config.action_l2_regularization * F.mse_loss(a / action_scale, th.zeros_like(a))

    actor_optimizer.zero_grad()
    actor_loss.backward()
      
    # Grad clipping
    if config.grad_norm_clipping > 0.:	
      raise
      for p in actor_parameters:
        clip_coef = config.grad_norm_clipping / (1e-6 + th.norm(p.grad.detach()))
        if clip_coef < 1:
          p.grad.detach().mul_(clip_coef)
    if config.grad_value_clipping > 0.:
      raise
      th.nn.utils.clip_grad_value_(actor_parameters, config.grad_value_clipping)
      
    actor_optimizer.step()

    for p in critic_parameters:
      p.requires_grad = True


class DDPG(OffPolicyAgent):
  u"""Deep Deterministic Policy Gradient (DDPG).

  Arguments:

    env (Env): The vectorized training environments.
    test_env (Env): The vectorized evaluation environments.
    observation_normalizer (ObservationNormalizer): 
      Normalize observations.
    buffer (Buffer): The replay buffer.
    feature_extractor (Extractor): Common feature extractor.
    policy (): The policy model to use.
    action_noise (ActionNoise): Add noises to actions.
    lr (float): Learning rate for the optimizer..
    learning_starts (int): How many steps the agent takes to 
      collect transitions before training starts.
    batch_size (int): Minibatch size for replay buffer sampling.
    tau (float): The polyak update coefficient.
    gamma (float): The discount factor.
    train_every_n_steps (int): Train this agent every these steps.
      It must be a multiple of `env.n_envs`.
    gradient_steps (int): How many gradient steps to take.
    logging_window (int): Window size for logging, specifying 
      the number of episodes to average.
    verbose (int): Verbosity.

  References:
    - http://proceedings.mlr.press/v32/silver14.pdf
    - https://arxiv.org/abs/1509.02971
    - https://spinningup.openai.com/en/latest/algorithms/ddpg.html

  """

  def __init__(self,
               config,
               env,
               test_env,
               observation_normalizer: ObservationNormalizer,
               buffer: Buffer,
               feature_extractor: Extractor,
               policy: DDPGPolicy,
               action_noise: ActionNoise,
               lr: float,
               learning_starts: int,
               batch_size: int = 256,
               tau: float = 0.005,
               gamma: float = 0.99,
               train_every_n_steps: int = -1,
               gradient_steps: int = 1,
               logging_window: int = 30,
               verbose: int = 0):
    super().__init__(config, 
                     env, 
                     test_env, 
                     observation_normalizer,
                     buffer, 
                     feature_extractor, 
                     policy, 
                     action_noise, 
                     lr,
                     learning_starts,
                     batch_size,
                     tau,
                     gamma,
                     train_every_n_steps,
                     gradient_steps,
                     logging_window,
                     verbose)

  @overrides
  def update(self, gradient_steps: int):
    for _ in range(gradient_steps):
      self._policy.optimize_batch(*self.get_transitions())
      for target, model in self._policy.targets_and_models:
        soft_update(target, model, self._tau)