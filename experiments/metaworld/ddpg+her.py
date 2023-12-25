

import argparse
import gym
import numpy as np
import torch as th
import torch.nn as nn

from collections import OrderedDict

from rldev.agents.common.normalizer import *
from rldev.agents.common.action_noise import *
from rldev.agents.ddpg import DDPG, DDPGPolicy
from rldev.agents.policy.ac import Actor, Critic
from rldev.buffers.her import OnlineHERBuffer
from rldev.configs import push_args, best_slide_config
from rldev.environments import EnvModule
from rldev.utils import torch as ptu
from rldev.utils.nn import FCBody

from rldev.agents.core.bpref import utils


def main(config):

  if config.gamma < 1.: config.clip_target_range = (np.round(-(1 / (1 - config.gamma)), 2), 0.)
  if config.gamma == 1: config.clip_target_range = (np.round(-config.env_max_step - 5, 2), 0.)

  th.set_num_threads(min(4, config.num_envs))
  th.set_num_interop_threads(min(4, config.num_envs))

  def make(cfg):
    from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
    from gym.wrappers.time_limit import TimeLimit
    from gym.spaces import Dict, Box
    from rldev.agents.core.bpref.rlkit.envs.wrappers import NormalizedBoxEnv

    env_name = cfg.env.replace('metaworld_','')
    env_name += "-goal-observable"
    env_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_name]
    
    env = env_cls()    
    env._freeze_rand_vec = False
    env._set_task_called = True
    env.seed(cfg.seed)
    env = TimeLimit(NormalizedBoxEnv(env), env.max_path_length)

    class X:

      def __init__(self, env):
        self.env = env

      @property
      def observation_space(self):
        space = self.env.observation_space
        low = space.low
        high = space.high
        return Dict(
          {"achieved_goal": Box(low=low[4:7], high=high[4:7]),
           "desired_goal": Box(low=low[-3:], high=high[-3:]),
           "observation": Box(low=np.hstack([low[:4], low[7:-3]]), high=np.hstack([high[:4], high[7:-3]]))})

      def dict_observation(self, x):
        opos = x[4:7] # object position (achieved)
        gpos = x[-3:] # goal position (desired)
        return OrderedDict(
          [("achieved_goal", np.copy(opos)),
           ("desired_goal", np.copy(gpos)),
           ("observation", np.copy(np.hstack([x[:4], x[7:-3]])))])

      def compute_reward(self, achieved_goal, desired_goal, info):
        observation = info["dict"]
        batched = False
        n = None
        for key, x in observation.items():
          if len(x.shape) > 1:
            batched = True
            n = x.shape[0]
            break
        def process(dict_obs):
          return np.concatenate(
            [dict_obs["observation"][:4], 
             dict_obs["achieved_goal"][:], 
             dict_obs["observation"][4:], 
             dict_obs["desired_goal"][:]], axis=-1)          
        if not batched:
          return self.env.env.compute_reward(None, process(info["dict"]))[0]
        rewards = []
        for i in range(n):
          obs = OrderedDict([(key, observation[key][i]) for key in observation])
          # SawyerButtonPressEnvV2 doesn't require action for reward computation.
          rewards.append(self.env.env.compute_reward(None, process(obs))[0])
        return np.array(rewards)

      def reset(self):
        return self.dict_observation(self.env.reset())
      
      def step(self, action):
        x, *extra = self.env.step(action)
        return self.dict_observation(x), *extra

      def __getattr__(self, key):
        return self.env.__getattr__(key)

    return X(env)

  env = lambda: make(config)

  train_env = EnvModule(env, num_envs=config.num_envs, seed=config.seed)
  test_env = EnvModule(env, num_envs=config.num_envs, name='test_env', seed=config.seed + 1138)

  # print(config.modalities, config.goal_modalities)
  # print(train_env.observation_space.spaces["achieved_goal"].dtype)
  # print(config.get("her"))
  # exit(0)

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
               OnlineHERBuffer,
               observation_normalizer,
               action_noise)
  
  num_eps = max(config.num_envs * 3, 10)
  agent.test(num_eps)
  agent.run(config.epoch_steps, num_eps)


if __name__ == '__main__':

  env_choices = ["button-press-v2"]

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
  parser.add_argument("--episode_steps",
    default=50, type=int, help="length of an episode in steps")
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
