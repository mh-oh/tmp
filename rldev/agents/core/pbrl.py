
import copy
import numpy as np
import os
import time

from abc import *
from collections import deque
from overrides import overrides

from rldev.agents.core import Agent
from rldev.agents.core.bpref import utils


class DummyLogger:

  def __init__(self, *args, **kwargs):
    ...

  def log(self, *args, **kwargs):
    ...

  def log_histogram(self, *args, **kwargs):
    ...

  def log_param(self, *args, **kwargs):
    ...

  def dump(self, *args, **kwargs):
    ...


class PbRLAgent(Agent, metaclass=ABCMeta):

  @overrides
  def setup_logger(self): return DummyLogger()

  def __init__(self,
               config,
               env,
               test_env,
               policy,
               buffer,
               reward_model):
    super().__init__(config,
                     env,
                     test_env,
                     policy)

    self._buffer = buffer
    self._reward_model = reward_model

    self._feedbacks = 0
    self._labeled_feedbacks = 0
    
    self._step = 0
    self._episode = 0

    self._n_envs = n_envs = self._env.num_envs
    self._episode_pseudo_return, self._done = np.zeros((n_envs,)), np.ones((n_envs,), dtype=bool)
    self._episode_success = np.zeros((n_envs,))
    self._episode_return = np.zeros((n_envs,))
    self._episode_step = np.zeros((n_envs,))

    self._episode_returns = []
    self._episode_steps = []
    self._episode_successes = []

    # store train returns of recent 10 episodes
    self._avg_train_true_return = deque([], maxlen=10) 
    self._start_time = time.time()

    self._interact_count = 0

  @property
  def config(self):
    return self._config
  
  @property
  def logger(self):
    return self._logger

  def run(self, test_episodes: int):

    epoch_length = self.config.eval_frequency
    self.obs = self._env.reset()
    for epoch in range(self._training_steps // epoch_length):
      self.train(epoch_length)
      self.logger.log('eval/episode', self._episode, self._step)
      self.test(test_episodes)

    self._policy.save(self.workspace, self._step)
    self._reward_model.save(self.workspace, self._step)


  def train(self, epoch_length):

    for _ in range(epoch_length // self._n_envs):

      if np.any(self._done):
        self._episode_returns += list(self._episode_return[self._done])
        self._episode_successes += list(self._episode_success[self._done])
        self._episode_steps += list(self._episode_step[self._done])
        self._episode_return[self._done] = 0
        self._episode_step[self._done] = 0
        self._episode += np.sum(self._done)
        self._done[self._done] = False

      self.logger.log("train/episode", self._episode, self._step)

      # sample action for data collection
      assert self.obs.shape == (1, 39)
      if self._step < self.config.num_seed_steps:
        action = np.array([self._env.action_space.sample() for _ in range(self._env.num_envs)])
      else:
        with utils.eval_mode(self._policy):
          action = self._policy.act(self.obs, sample=True)
      assert action.shape == (1, 4)
      self.optimize_reward_model()
      self.optimize_policy()

      next_obs, reward, self._done, info = self._env.step(action)
      # As the VecEnv resets automatically, new_obs is already the
      # first observation of the next episode
      __next_obs = copy.deepcopy(next_obs)
      for i, done in enumerate(self._done):
        terminal = info[i].get("terminal_observation")
        if done and terminal is not None:
          __next_obs[i] = terminal

      pseudo_reward = self._reward_model.r_hat(np.concatenate([self.obs, action], axis=-1))[..., 0]

      # allow infinite bootstrap
      # self._done = float(self._done)
      done_no_max = copy.deepcopy(self._done)
      for i in range(self._n_envs):
        if self._episode_step[i] + 1 == self._env._max_episode_steps:
          done_no_max[i] = False
      done_no_max = done_no_max.astype(float)
      
      for i in range(self._n_envs):
        self._episode_success[i] = max(self._episode_success[i], info[i]["success"])
        self._episode_pseudo_return[i] += pseudo_reward
        self._episode_return[i] += reward
        self._episode_step[i] += 1
          
      # adding data to the reward training data
      self.process_experience(self.obs,
                              action,
                              reward,
                              pseudo_reward,
                              __next_obs,
                              self._done,
                              done_no_max)

      self.obs = next_obs
      self._step += self._n_envs
      self._interact_count += self._n_envs

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
                         done_no_max):
    self._buffer.add(observation, 
                     action, 
                     pseudo_reward, 
                     next_observation, 
                     done,
                     done_no_max)
    self._reward_model.add_data(observation, 
                                action, 
                                reward, 
                                done)

  def test(self,
           episodes: int):

    episode_returns = []
    episode_successes = []

    env = self._test_env
    
    while len(episode_returns) < episodes:
      obs = env.reset()
      done = np.zeros((self._n_envs,))
      episode_success = np.zeros((self._n_envs,))
      episode_return = np.zeros((self._n_envs,))

      while not np.all(done):
        with utils.eval_mode(self._policy):
          action = self._policy.act(obs, sample=False)
        obs, reward, done, info = env.step(action)
        for i in range(self._n_envs):
          if not done[i]:
            episode_return[i] += reward[i]
            episode_success[i] = max(episode_success[i], info[i]["success"])
      
      episode_returns.extend(episode_return)
      episode_successes.extend(episode_success)
    
    self.logger.log("eval/episode_return", 
                    np.mean(episode_returns), self._step)
    self.logger.log("eval/success_rate",
                    np.mean(episode_successes), self._step)
    self.logger.dump(self._step)