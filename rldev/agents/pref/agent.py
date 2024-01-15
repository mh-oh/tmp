
import copy
import numpy as np
import os
import time

from abc import *
from collections import deque
from overrides import overrides
from pathlib import Path

from rldev.agents.core import Agent
from rldev.agents.pref import utils
from rldev.logging import DummyLogger
from rldev.utils import torch as thu
from rldev.utils.env import get_success_info
from rldev.utils.structure import recursive_get


class PbRLAgent(Agent, metaclass=ABCMeta):

  @overrides
  def setup_logger(self): return DummyLogger(self)

  def __init__(self,
               config,
               env,
               test_env,
               policy,
               feature_extractor,
               buffer,
               reward_model,
               window=10):
    super().__init__(config,
                     env,
                     test_env,
                     policy,
                     feature_extractor)

    self._buffer = buffer(self)
    self._reward_model = reward_model(self)
    self._n_envs = n_envs = self._env.num_envs

    # What is this for? Probabily, $K$ in the paper?
    self._interact_count = 0

    self._logger.define("train/epoch",
                        "train/episode",
                        "train/episode_steps",
                        "train/success_rate",
                        "train/return",
                        "train/sparse_return",
                        "train/progress",
                        "train/pseudo_return",
                        "train/feedbacks",
                        "train/labeled_feedbacks",
                        "test/success_rate",
                        "test/return",
                        "test/sparse_return",
                        "test/progress")

    u"""Training records."""

    self._step = 0
    self._episode = 0

    self._done = np.ones((n_envs,), dtype=bool)
    self._episode_step = np.zeros((n_envs,))
    self._episode_success = np.zeros((n_envs,))
    self._episode_progress = np.full((n_envs,), np.inf)
    self._episode_return = np.zeros((n_envs,))
    self._episode_sparse_return = np.zeros((n_envs,))
    self._episode_pseudo_return = np.zeros((n_envs,))

    # We keep track of recent `window` episodes for aggregation.
    self._episode_steps = deque([], maxlen=window)
    self._episode_successes = deque([], maxlen=window)
    self._episode_progresses = deque([], maxlen=window)
    self._episode_returns = deque([], maxlen=window)
    self._episode_sparse_returns = deque([], maxlen=window)
    self._episode_pseudo_returns = deque([], maxlen=window)

    # Number of human feedbacks until current step.
    self._feedbacks = 0
    self._labeled_feedbacks = 0

    self._log_every_n_steps = config.log_every_n_steps

  @property
  def config(self):
    return self._config
  
  @property
  def logger(self):
    return self._logger

  def run(self, test_episodes: int):

    epoch_length = self.config.test_every_n_steps
    self.obs = self._env.reset()
    for epoch in range(self._training_steps // epoch_length):
      self.logger.log("train/epoch", epoch, self._step)
      self.train(epoch_length)
      self.test(test_episodes)
      self.save()

  def train(self, epoch_length):

    env = self._env
    for _ in range(epoch_length // self._n_envs):
      self.process_episodic_records(self._done)
      if np.any(self._done):
        self._done[self._done] = False

      # sample action for data collection
      if self._step < self.config.num_seed_steps:
        action = np.array([env.action_space.sample() for _ in range(env.num_envs)])
      else:
        with utils.eval_mode(self._policy):
          obs = self._feature_extractor(self.obs)
          action = self._policy.act(obs, sample=True)
      assert action.ndim == 2
      self.optimize_reward_model()
      self.optimize_policy()

      maybe_next_observation, reward, self._done, info = env.step(action)
      # As the VecEnv resets automatically, new_obs is already the
      # first observation of the next episode
      next_observation = copy.deepcopy(maybe_next_observation)
      for i, done in enumerate(self._done):
        terminal = info[i].get("terminal_observation")
        if done and terminal is not None:
          next_observation[i] = terminal

      pseudo_reward = thu.numpy(self._reward_model.r_hat(self.obs, action))[..., 0]

      # allow infinite bootstrap
      # self._done = float(self._done)
      done_no_max = copy.deepcopy(self._done)
      for i in range(self._n_envs):
        if self._episode_step[i] + 1 == env._max_episode_steps:
          done_no_max[i] = False
      done_no_max = done_no_max.astype(float)
      
      progress = env.envs[0].compute_progress
      for i in range(self._n_envs):
        success = get_success_info(info[i])
        if success is not None:
          self._episode_success[i] = max(self._episode_success[i], success)
        self._episode_progress[i] = progress(recursive_get(i, self.obs))
        self._episode_pseudo_return[i] += pseudo_reward ################################
        self._episode_return[i] += reward[i] ###########################
        self._episode_sparse_return[i] += info[i]["sparse_reward"]
        self._episode_step[i] += 1
          
      # adding data to the reward training data
      self.process_experience(self.obs,
                              action,
                              reward,
                              pseudo_reward,
                              next_observation,
                              self._done,
                              done_no_max,
                              info)

      self.obs = maybe_next_observation
      self._step += self._n_envs
      self._interact_count += self._n_envs

  @abstractmethod
  def process_episodic_records(self, done):

    if np.any(done):
      self._episode += np.sum(done)
      self._episode_steps.extend(self._episode_step[done])
      self._episode_successes.extend(self._episode_success[done])
      self._episode_progresses.extend(self._episode_progress[done])
      self._episode_returns.extend(self._episode_return[done])
      self._episode_sparse_returns.extend(self._episode_sparse_return[done])
      self._episode_pseudo_returns.extend(self._episode_pseudo_return[done])
      self._episode_success[done] = 0
      self._episode_return[done] = 0
      self._episode_sparse_return[done] = 0
      self._episode_pseudo_return[done] = 0
      self._episode_step[done] = 0

    if self._step % self._log_every_n_steps < self._n_envs:
      self.logger.log("train/episode", self._episode, self._step)
      self.logger.log("train/episode_steps", np.mean(self._episode_steps), self._step)
      self.logger.log("train/success_rate", np.mean(self._episode_successes), self._step)
      self.logger.log("train/progress", np.mean(self._episode_progresses), self._step)
      self.logger.log("train/return", np.mean(self._episode_returns), self._step)
      self.logger.log("train/sparse_return", np.mean(self._episode_sparse_returns), self._step)
      self.logger.log("train/pseudo_return", np.mean(self._episode_pseudo_returns), self._step)
      self.logger.log("train/feedbacks", self._feedbacks, self._step)
      self.logger.log("train/labeled_feedbacks", self._labeled_feedbacks, self._step)

  @abstractmethod
  def optimize_reward_model(self):
    ...

  @abstractmethod
  def optimize_policy(self):
    ...

  def process_experience(self,
                         observation,
                         action,
                         reward,
                         pseudo_reward,
                         next_observation,
                         done,
                         done_no_max,
                         info):

    self._buffer.add(observation, 
                     action, 
                     pseudo_reward, 
                     next_observation, 
                     done_no_max)
    self._reward_model.add(observation,
                           action,
                           reward,
                           next_observation,
                           done)

  def test(self,
           episodes: int):

    episode_returns = []
    episode_sparse_returns = []
    episode_progresses = []
    episode_successes = []

    env = self._test_env
    while len(episode_returns) < episodes:
      obs = env.reset()

      done = np.zeros((self._n_envs,))
      episode_success = np.zeros((self._n_envs,))
      episode_progress = np.full((self._n_envs,), np.inf)
      episode_return = np.zeros((self._n_envs,))
      episode_sparse_return = np.zeros((self._n_envs,))

      while not np.all(done):
        with utils.eval_mode(self._policy):
          action = self._policy.act(
            self._feature_extractor(obs), sample=False)
        
        obs, reward, done, info = env.step(action)
        for i in range(self._n_envs):
          progress = env.envs[0].compute_progress
          if done[i]:
            episode_progress[i] = progress(recursive_get(i, obs))
          else:
            episode_return[i] += reward[i]
            episode_sparse_return[i] += info[i]["sparse_reward"]
            success = get_success_info(info[i])
            if success is not None:
              episode_success[i] = max(episode_success[i], success)
      
      episode_returns.extend(episode_return)
      episode_sparse_returns.extend(episode_sparse_return)
      episode_progresses.extend(episode_progress)
      episode_successes.extend(episode_success)
    
    self.logger.log("test/return", 
                    np.mean(episode_returns), self._step)
    self.logger.log("test/sparse_return", 
                    np.mean(episode_sparse_returns), self._step)
    self.logger.log("test/progress",
                    np.mean(episode_progresses), self._step)
    self.logger.log("test/success_rate",
                    np.mean(episode_successes), self._step)