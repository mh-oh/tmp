

import argparse
import gym
import gymnasium
import numpy as np
import torch as th
import torch.nn as nn

from rldev.agents.common.normalizer import *
from rldev.agents.common.action_noise import *
from rldev.agents.ddpg import DDPG, DDPGPolicy
from rldev.agents.policy.ac import Actor, Critic
from rldev.buffers.her import HindsightBuffer
from rldev.configs import push_args, best_slide_config
from rldev.environments import EnvModule
from rldev.environments.wrappers import GymApi
from rldev.utils import torch as ptu
from rldev.utils.nn import FCBody


def main(config):

  if config.gamma < 1.: config.clip_target_range = (np.round(-(1 / (1 - config.gamma)), 2), 0.)
  if config.gamma == 1: config.clip_target_range = (np.round(-config.env_max_step - 5, 2), 0.)

  th.set_num_threads(min(4, config.num_envs))
  th.set_num_interop_threads(min(4, config.num_envs))

  assert gymnasium.envs.registry.get(config.env) is not None
  env = lambda: GymApi(gymnasium.make(config.env))

  train_env = EnvModule(env, num_envs=config.num_envs, seed=config.seed)
  test_env = EnvModule(env, num_envs=config.num_envs, name='test_env', seed=config.seed + 1138)

  e = test_env
  actor = Actor(FCBody(e.state_dim + e.goal_dim, config.policy_layers, nn.LayerNorm), e.action_dim, e.max_action).to(ptu.device())
  critic = Critic(FCBody(e.state_dim + e.goal_dim + e.action_dim, config.policy_layers, nn.LayerNorm), 1).to(ptu.device())

  policy = (
    lambda agent: 
      DDPGPolicy(agent, 
                 train_env.max_action,
                 actor,
                 config.actor_lr,
                 config.actor_weight_decay,
                 critic,
                 config.critic_lr,
                 config.critic_weight_decay))
  
  buffer = (
    lambda agent:
      HindsightBuffer(agent,
                      train_env.num_envs,
                      config.replay_size,
                      train_env.observation_space,
                      train_env.action_space,
                      config.her))

  observation_normalizer = (
    lambda agent: Normalizer(agent, MeanStdNormalizer()))
  action_noise = (
    lambda agent: 
      ContinuousActionNoise(
        agent, 
        GaussianProcess, 
        std=ConstantSchedule(config.action_noise)))

  if e.goal_env:
    config.never_done = True  # NOTE: This is important in the standard Goal environments, which are never done

  agent = DDPG(config,
               train_env,
               test_env,
               policy,
               buffer,
               observation_normalizer,
               action_noise)
  
  num_eps = max(config.num_envs * 3, 10)
  agent.test(num_eps)
  agent.run(config.epoch_steps, num_eps)


if __name__ == '__main__':

  env_choices = ["PointMaze_UMaze-v3",
                 "PointMaze_UMazeDense-v3",
                 "PointMaze_Open-v3",
                 "PointMaze_OpenDense-v3"]

  parser = argparse.ArgumentParser()
  parser.add_argument("--run", 
    required=True, type=str, help="name of this run")
  parser.add_argument("--seed",
    required=True, type=int, help="seed of this run")
  parser.add_argument("--env", 
    required=True, type=str, choices=env_choices,
    help="name of environment")
  parser.add_argument("--num_envs", 
    default=1, type=int, help="the number of envrionments for vectorization")
  parser.add_argument("--steps",
    default=1000000, type=int, help="training steps")
  parser.add_argument("--epoch_steps", 
    default=5000, type=int, help="length of an epoch in steps")
  parser.add_argument("--policy_layers",
    nargs='+', default=(512, 512, 512), type=int, help="hidden layers for actor/critic")
  args = parser.parse_args()

  config = best_slide_config()
  config = push_args(config, args)
  from pprint import pprint
  pprint(config)

  import subprocess, sys
  config.launch_command = sys.argv[0] + ' ' + subprocess.list2cmdline(sys.argv[1:])

  main(config)
