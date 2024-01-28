
import numpy as np
import torch as th
import torch.nn as nn

from rldev.agents.observation_normalizer import MeanStdNormalizer
from rldev.agents.action_noise import GaussianActionNoise
from rldev.agents.ddpg import DDPG, DDPGPolicy
from rldev.agents.policy.ac import Actor, Critic
from rldev.buffers.hindsight import HindsightBuffer
from rldev.environments import make_vec_env
from rldev.feature_extractor import Combine
from rldev.launcher import configure
from rldev.utils import torch as ptu
from rldev.utils.nn import FCBody


@configure
def main(conf):

  if conf.gamma < 1.: conf.clip_target_range = (np.round(-(1 / (1 - conf.gamma)), 2), 0.)
  if conf.gamma == 1: conf.clip_target_range = (np.round(-conf.env_max_step - 5, 2), 0.)

  th.set_num_threads(min(4, conf.num_envs))
  th.set_num_interop_threads(min(4, conf.num_envs))

  env = make_vec_env("FetchPush-v2", seed=1, n_envs=8)
  test_env = make_vec_env("FetchPush-v2", seed=1000, n_envs=8)

  e = test_env
  actor = Actor(FCBody(e.state_dim + e.goal_dim, conf.policy_layers), e.action_dim, e.max_action).to(ptu.device())
  critic = Critic(FCBody(e.state_dim + e.goal_dim + e.action_dim, conf.policy_layers), 1).to(ptu.device())

  feature_extractor = Combine(env.observation_space,
                              keys=["observation",
                                    "desired_goal"])
  print(feature_extractor.feature_space)
  policy = (
    lambda agent: 
      DDPGPolicy(agent, 
                 env.max_action,
                 actor,
                 conf.actor_lr,
                 conf.actor_weight_decay,
                 critic,
                 conf.critic_lr,
                 conf.critic_weight_decay))

  def compute_reward(observation, action, next_observation):
    return env.compute_reward(next_observation["achieved_goal"], 
                              observation["desired_goal"], 
                              {})


  buffer = (
    HindsightBuffer(env.num_envs,
                    conf.replay_size,
                    env.observation_space,
                    env.action_space,
                    compute_reward,
                    conf.her))

  observation_normalizer = MeanStdNormalizer(env.observation_space)
  action_noise = GaussianActionNoise(mean=0.0, stddev=conf.action_noise)

  if e.goal_env:
    conf.never_done = True  # NOTE: This is important in the standard Goal environments, which are never done

  assert conf.actor_lr == conf.critic_lr
  agent = DDPG(conf,
               env,
               test_env,
               observation_normalizer,
               buffer,
               feature_extractor,
               policy,
               action_noise,
               conf.actor_lr,
               learning_starts=5000,
               batch_size=2000,
               tau=0.05,
               gamma=0.98,
               train_every_n_steps=8,
               gradient_steps=4)
  
  num_eps = max(conf.num_envs * 3, 10)
  agent.test(num_eps)
  # agent.run(conf.epoch_steps, num_eps)
  agent.learn(conf.steps, 100, num_eps)


if __name__ == "__main__":
  main()