
import numpy as np
import torch as th
import torch.nn as nn

from rldev.agents.common.normalizer import *
from rldev.agents.common.action_noise import *
from rldev.agents.ddpg import DDPG, DDPGPolicy
from rldev.agents.policy.ac import Actor, Critic
from rldev.buffers.her import HindsightBuffer
from rldev.environments import EnvModule, create_env_by_name
from rldev.launcher import configure
from rldev.utils import torch as ptu
from rldev.utils.nn import FCBody


@configure("rldev.experiments")
def main(conf):

  if conf.gamma < 1.: conf.clip_target_range = (np.round(-(1 / (1 - conf.gamma)), 2), 0.)
  if conf.gamma == 1: conf.clip_target_range = (np.round(-conf.env_max_step - 5, 2), 0.)

  th.set_num_threads(min(4, conf.num_envs))
  th.set_num_interop_threads(min(4, conf.num_envs))

  env = lambda: create_env_by_name(conf.env)

  train_env = EnvModule(env, num_envs=conf.num_envs, seed=conf.seed)
  test_env = EnvModule(env, num_envs=conf.num_envs, name='test_env', seed=conf.seed + 1138)

  e = test_env
  actor = Actor(FCBody(e.state_dim + e.goal_dim, conf.policy_layers, nn.LayerNorm), e.action_dim, e.max_action).to(ptu.device())
  critic = Critic(FCBody(e.state_dim + e.goal_dim + e.action_dim, conf.policy_layers, nn.LayerNorm), 1).to(ptu.device())

  policy = (
    lambda agent: 
      DDPGPolicy(agent, 
                 train_env.max_action,
                 actor,
                 conf.actor_lr,
                 conf.actor_weight_decay,
                 critic,
                 conf.critic_lr,
                 conf.critic_weight_decay))
  
  buffer = (
    lambda agent:
      HindsightBuffer(agent,
                      train_env.num_envs,
                      conf.replay_size,
                      train_env.observation_space,
                      train_env.action_space,
                      conf.her))

  observation_normalizer = (
    lambda agent: Normalizer(agent, MeanStdNormalizer()))
  action_noise = (
    lambda agent: 
      ContinuousActionNoise(
        agent, 
        GaussianProcess, 
        std=ConstantSchedule(conf.action_noise)))

  if e.goal_env:
    conf.never_done = True  # NOTE: This is important in the standard Goal environments, which are never done

  agent = DDPG(conf,
               train_env,
               test_env,
               policy,
               buffer,
               observation_normalizer,
               action_noise)
  
  num_eps = max(conf.num_envs * 3, 10)
  agent.test(num_eps)
  agent.run(conf.epoch_steps, num_eps)


if __name__ == "__main__":
  main()