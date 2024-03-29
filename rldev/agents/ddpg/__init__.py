
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from abc import *
from overrides import overrides

from rldev.agents import ActionNoise, ObservationNormalizer
from rldev.agents.core import OffPolicyAgent
from rldev.agents.ddpg.policies import Policy
from rldev.buffers.basic import Buffer
from rldev.feature_extractor import Extractor
from rldev.utils.nn import soft_update


class DDPGPolicy(Policy):

  def __init__(self, 
               max_action: float, 
               pi: nn.Module, 
               pi_lr: float, 
               pi_weight_decay: float, 
               qf: nn.Module, 
               qf_lr: float, 
               qf_weight_decay: float):
    super().__init__(max_action, 
                     pi, 
                     pi_lr, 
                     pi_weight_decay, 
                     qf, 
                     qf_lr, 
                     qf_weight_decay)

  @overrides
  def optimize_batch(self,
                     observation,
                     action,
                     reward,
                     next_observation,
                     done):

    action_scale = self._max_action

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
      gamma = 0.98
      target = (reward + (gamma * (1. - done)) * q_next)
      clip_target_range = (np.round(-(1 / (1 - gamma)), 2), 0.)
      target = th.clamp(target, *clip_target_range)

    # if config.opt_steps % 1000 == 0:
    #   logger.add_histogram('Optimize/Target_q', target)
    
    q = critic(observation, action)
    critic_loss = F.mse_loss(q, target)

    critic_optimizer.zero_grad()
    critic_loss.backward()
      
    # Grad clipping

    critic_optimizer.step()

    for p in critic_parameters:
      p.requires_grad = False

    a = actor(observation)
      
    actor_loss = -critic(observation, a)[:, -1].mean()
    # action l2 regularization
    actor_loss += 1e-2 * F.mse_loss(a / action_scale, th.zeros_like(a))

    actor_optimizer.zero_grad()
    actor_loss.backward()
      
    # Grad clipping
      
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