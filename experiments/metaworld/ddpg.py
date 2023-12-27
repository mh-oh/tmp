

import argparse
import gym
import numpy as np
import torch as th
import torch.nn as nn

from abc import *
from collections import OrderedDict
from gym.spaces import Box, Dict
from gym.wrappers.time_limit import TimeLimit

from rldev.agents.common.normalizer import *
from rldev.agents.common.action_noise import *
from rldev.agents.ddpg import DDPG, DDPGPolicy
from rldev.agents.policy.ac import Actor, Critic
from rldev.buffers.basic import DictBuffer
from rldev.configs import push_args, best_slide_config
from rldev.environments import EnvModule
from rldev.utils import torch as ptu
from rldev.utils.env import observation_spec
from rldev.utils.nn import FCBody


def requires_box_observation(observation):
  ...


def goal_distance(x, y):
  assert x.shape == y.shape
  return np.linalg.norm(x - y, axis=-1)


class BoxGoalEnv:
  ...

  def __init__(self, env, mode="sparse"):

    self._env = env
    self._reward_mode = mode
    self._box_observation_space = env.observation_space
    self._box_spec = observation_spec(self._box_observation_space)

    def box(index):
      space = self._box_observation_space
      return Box(
        low=space.low[index], high=space.high[index], dtype=space.dtype)
    self._dict_observation_space = Dict(
      [(key, box(self.get_index(key))) for key in self.observation_keys])

  @property
  def observation_keys(self):
    return ["observation", "achieved_goal", "desired_goal"]

  @property
  def box_observation_space(self):
    return self._box_observation_space

  @property
  def dict_observation_space(self):
    return self._dict_observation_space

  @property
  def observation_space(self):
    return self.dict_observation_space

  def get_index(self, key):
    if key not in self.observation_keys:
      raise ValueError(
        f"key should be one of {self.observation_keys}")
    return self.index(key)

  @abstractmethod
  def index(self, key):
    ...
  
  def to_dict_observation(self, box_observation):

    if not isinstance(box_observation, np.ndarray):
      raise ValueError(f"")

    dict_observation = OrderedDict()
    for key in self.observation_keys:
      dict_observation[key] = box_observation[self.get_index(key)]

    return dict_observation

  def to_box_observation(self, dict_observation):    
    
    if not isinstance(dict_observation, (dict, OrderedDict)):
      raise ValueError(f"")

    spec = self._box_spec
    shapes = []
    for key in self.observation_keys:
      shapes.append(dict_observation[key].shape[:-len(spec.shape)])
    if len(set(shapes)) != 1:
      raise ValueError()
    box_observation = np.zeros(shapes[0] + spec.shape, dtype=spec.dtype)
    for key in self.observation_keys:
      box_observation[..., self.get_index(key)] = dict_observation[key]

    return box_observation

  def __getattr__(self, name):
    return getattr(self._env, name)

  def observation(self, box_observation):

    dict_observation = self.to_dict_observation(box_observation)
    dtype = self._box_spec.dtype
    assert (box_observation.astype(dtype) 
            != self.to_box_observation(dict_observation)).sum() <= 0
    return dict_observation

  def reset(self):
    return self.observation(self._env.reset())
  
  def step(self, action):
    box_observation, *extra = self._env.step(action)
    return self.observation(box_observation), *extra
  
  def compute_reward(self, achieved, desired, info):
    raise
    
    mode = self._reward_mode
    if mode == "dense":
      actions = info["action"]
      next_observations = self.to_box_observation(
        info["next_observation"])
      rewards = []
      for action, next_observation in zip(actions, next_observations):
        rewards.append(
          self._env.compute_reward(action, 
                                  next_observation)[0])
      return np.array(rewards)

    # Compute distance between goal and the achieved goal.
    d = goal_distance(achieved, desired)
    if mode == "sparse":
      return -(d > 0.05).astype(np.float32)
    elif mode == "distance":
      return -d
    
    raise ValueError(f"...")


class PushV2(BoxGoalEnv):

  @overrides
  def index(self, key):

    shape = self._box_observation_space.shape
    dim = np.prod(shape)
    if key == "desired_goal":
      return [36, 37, 38]
    elif key == "achieved_goal":
      return [4, 5, 6]
    elif key == "observation":
      return np.delete(
        np.arange(dim).reshape(shape), self.index("desired_goal"))


def main(config):

  if config.gamma < 1.: config.clip_target_range = (np.round(-(1 / (1 - config.gamma)), 2), 0.)
  if config.gamma == 1: config.clip_target_range = (np.round(-config.env_max_step - 5, 2), 0.)

  th.set_num_threads(min(4, config.num_envs))
  th.set_num_interop_threads(min(4, config.num_envs))

  from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
  cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[config.env]
  env = PushV2(cls())
  env = TimeLimit(env, env.max_path_length)
  env_fn = lambda: env

  train_env = EnvModule(env_fn, num_envs=config.num_envs, seed=config.seed)
  test_env = EnvModule(env_fn, num_envs=config.num_envs, name='test_env', seed=config.seed + 1138)

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
      DictBuffer(agent,
                 train_env.num_envs,
                 config.replay_size,
                 train_env.observation_space,
                 train_env.action_space))

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

  env_choices = ["push-v2-goal-observable",
                 ...]

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
