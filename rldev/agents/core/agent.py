
import numpy as np
import os
import sys
import time
import pickle

from abc import *
from collections import OrderedDict
from overrides import overrides
from pathlib import Path
from typing import *

from rldev.agents.core import Node
from rldev.logging import Logger
from rldev.utils.env import discounted_sum, debug_vectorized_experience, get_success_info
from rldev.utils.time import short_timestamp


class Agent(metaclass=ABCMeta):

  def setup_logger(self): return Logger(self)

  def __init__(self,
               config,
               env,
               test_env,
               policy):

    this_run_path = Path(os.path.realpath(sys.argv[0]))
    self._workspace = wdir = (Path(this_run_path.parent) 
                              / "data" 
                              / this_run_path.stem
                              / f"{config.run}")

    print(f"create working directory {wdir}")
    wdir.mkdir(parents=True, exist_ok=True)

    self._nodes = OrderedDict()

    self._config = config
    self._env = env
    self._test_env = test_env
    self._policy = policy(self)
    self._logger = self.setup_logger()

    self._training_steps = config.steps
    self._training = True

  def __setattr__(self, name: str, value: Any):
    if isinstance(value, Node):
      self._nodes[name] = value
    else:
      object.__setattr__(self, name, value)

  def __getattr__(self, name: str):
    if name in self._nodes:
      return self._nodes[name]
    return object.__getattribute__(self, name)

  @property
  def workspace(self):
    return self._workspace.resolve()
  
  @property
  def save_dir(self):
    return self.workspace / "agent"

  @property
  def config(self):
    return self._config
  
  @property
  def logger(self):
    return self._logger

  @abstractmethod
  def run(self, *args, **kwargs):
    ...
  
  def training_mode(self):
    self._training = True

  def evaluation_mode(self):
    self._training = False
  
  @property
  def training(self):
    return self._training
  
  @abstractmethod
  def save(self):

    # Save registered nodes.
    dir = self.save_dir
    for key, node in self._nodes.items():
      node.save(dir / key)
    
    # Save agent-specific data.
    with open(dir / "_config.pkl", "wb") as fout:
      pickle.dump(self._config, fout)
    with open(dir / "_training.pkl", "wb") as fout:
      pickle.dump(self._training, fout)

  @abstractmethod
  def load(self):

    # Load registered nodes.
    dir = self.save_dir
    for key, node in self._nodes.items():
      node.load(dir / key)
    
    # Load agent-specific data.
    with open(dir / "_config.pkl", "rb") as fin:
      self._config = pickle.load(fin)
    with open(dir / "_training.pkl", "rb") as fin:
      self._training = pickle.load(fin)


class OffPolicyAgent(Agent):
  
  def __init__(self, 
               config, 
               env, 
               test_env, 
               policy,
               buffer):
    super().__init__(config, env, test_env, policy)
    self._buffer = buffer(self)

    self._config.env_steps = 0
    self._config.opt_steps = 0

  @property
  def buffer(self):
    return self._buffer

  @overrides
  @abstractmethod
  def save(self): super().save()
  
  @overrides
  @abstractmethod
  def load(self): super().load()

  @overrides
  def run(self, 
          epoch_steps: int, 
          test_episodes: int, *args, **kwargs):

    for epoch in range(int(self._training_steps // epoch_steps)):
      elapsed = self.train(epoch_steps)
      self.logger.log_color(
        f"({epoch}) Training one epoch takes {elapsed:.2f} seconds.")
      _, elapsed = self.test(test_episodes)
      self.logger.log_color(
        f"({epoch}) Evaluation takes {elapsed:.2f} seconds.", color="yellow")
      self.logger.log_color(
        f"({epoch}) Saving...", color="crimson")
      self.save()

  def train(self, 
            epoch_steps: int, 
            render: bool = False, 
            dont_optimize: bool = False, 
            dont_train: bool = False):

    start = time.time()
    if not dont_train:
      self.training_mode()

    env = self._env
    state = env.state 

    for _ in range(epoch_steps // env.num_envs):
      action = self._policy(state)
      next_state, reward, done, info = env.step(action)

      state, experience = debug_vectorized_experience(state, action, next_state, reward, done, info)
      self.process_experience(experience)

      if render:
        time.sleep(0.02)
        env.render()
      
      for _ in range(env.num_envs):
        self._config.env_steps += 1
        if self._config.env_steps % self._config.optimize_every == 0 and not dont_optimize:
          self._config.opt_steps += 1
          self.optimize()
    
    # If using MEP prioritized replay, fit the density model
    if self._config.prioritized_mode == 'mep':
      self.prioritized_replay.fit_density_model()
      self.prioritized_replay.update_priority()
    
    return time.time() - start

  @abstractmethod
  def optimize(self):
    ...

  @abstractmethod
  def process_experience(self, experience):
    ...

  def test(self, 
           episodes: int, 
           any_success: bool = False):

    start = time.time()

    self.evaluation_mode()
    env = self._test_env
    num_envs = env.num_envs
    
    episode_rewards, episode_steps = [], []
    discounted_episode_rewards = []
    is_successes = []
    record_success = False

    while len(episode_rewards) < episodes:
      state = env.reset()

      dones = np.zeros((num_envs,))
      steps = np.zeros((num_envs,))
      is_success = np.zeros((num_envs,))
      ep_rewards = [[] for _ in range(num_envs)]

      while not np.all(dones):
        action = self._policy(state)
        state, reward, dones_, infos = env.step(action)

        for i, (rew, done, info) in enumerate(zip(reward, dones_, infos)):
          if dones[i]:
            continue
          ep_rewards[i].append(rew)
          steps[i] += 1
          if done:
            dones[i] = 1.
          success = get_success_info(info)
          if success is not None:
            record_success = True
            is_success[i] = max(success, is_success[i]) if any_success else success

      for ep_reward, step, is_succ in zip(ep_rewards, steps, is_success):
        if record_success:
          is_successes.append(is_succ)
        episode_rewards.append(sum(ep_reward))
        discounted_episode_rewards.append(discounted_sum(ep_reward, self._config.gamma))
        episode_steps.append(step)
    
    if len(is_successes):
      self._logger.add_scalar('Test/Success', np.mean(is_successes))
    self._logger.add_scalar('Test/Episode_rewards', np.mean(episode_rewards))
    self._logger.add_scalar('Test/Discounted_episode_rewards', np.mean(discounted_episode_rewards))
    self._logger.add_scalar('Test/Episode_steps', np.mean(episode_steps))

    return {'rewards': episode_rewards,
            'steps': episode_steps}, time.time() - start


class OnPolicyAgent(Agent):
  ...