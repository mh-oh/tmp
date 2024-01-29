
import numpy as np
import wandb

from abc import *
from collections import deque
from copy import deepcopy
from pathlib import Path

from rldev.agents import ActionNoise, ObservationNormalizer
from rldev.buffers.basic import Buffer
from rldev.feature_extractor import Extractor
from rldev.logging import WandbLogger, DummyLogger
from rldev.utils import torch as thu
from rldev.utils.env import get_success_info
from rldev.utils.typing import Obs, List, Dict, Any


class Agent(metaclass=ABCMeta):

  def setup_logger(self): return WandbLogger()

  def __init__(self,
               config,
               env,
               test_env,
               feature_extractor,
               policy,
               logging_window: int = 30):

    self._config = config
    self._env = env
    self._test_env = test_env
    self._feature_extractor = feature_extractor
    self._policy = policy
    self._logger = self.setup_logger() if logging_window > 0 else DummyLogger(self)

    self._training = True

  @property
  def workspace(self):
    if isinstance(self._logger, DummyLogger):
      raise AttributeError(
        "with 'logging=False', there is no workspace")
    return Path(wandb.run.dir)
  
  @property
  def save_dir(self):
    return self.workspace / "agent"

  @property
  def config(self):
    return self._config
  
  @property
  def logger(self):
    return self._logger
  
  def training_mode(self):
    self._training = True

  def evaluation_mode(self):
    self._training = False
  
  @property
  def training(self):
    return self._training


class OffPolicyAgent(Agent):
  u"""The base for off-policy agents (ex: SAC/TD3)

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

  Note the followings to understand the agent's behavior.

  * The agent starts learning on a call to `learn()`.
  * You should want to re-implement `update()` function that 
    updates network parameters (e.g., actor and critic).
  * The agent by default is evaluated periodically on a separate
    test environment `test_env`.

  """

  def __init__(self, 
               config, 
               env, 
               test_env, 
               observation_normalizer: ObservationNormalizer,
               buffer: Buffer,
               feature_extractor: Extractor, 
               policy,
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
                     feature_extractor, 
                     policy, 
                     logging_window)

    self._env = env
    self._test_env = test_env
    self._observation_normalizer = observation_normalizer
    self._buffer = buffer
    self._feature_extractor = feature_extractor
    self._policy = policy
    self._action_noise = action_noise
    self._lr = lr
    self._learning_starts = learning_starts
    self._batch_size = batch_size
    self._tau = tau
    self._gamma = gamma
    self._train_every_n_steps = train_every_n_steps
    self._gradient_steps = gradient_steps
    self._logging_window = logging_window
    self._verbose = verbose

    self.env_steps = 0
    self.opt_steps = 0

    ########################################################
    self._n_envs = n_envs = self._env.num_envs

    self._logger.define("train/epoch",
                        "train/episode",
                        "train/episode_steps",
                        "train/success_rate",
                        "train/return",
                        "test/success_rate",
                        "test/return")

    self._step = 0
    self._episode = 0

    self._done = np.ones((n_envs,), dtype=bool)
    self._episode_step = np.zeros((n_envs,))
    self._episode_success = np.zeros((n_envs,))
    self._episode_return = np.zeros((n_envs,))

    # We keep track of recent `window` episodes for aggregation.
    self._episode_steps = deque([], maxlen=logging_window)
    self._episode_successes = deque([], maxlen=logging_window)
    self._episode_returns = deque([], maxlen=logging_window)

    self._log_every_n_steps = config.log_every_n_steps

  @property
  def buffer(self):
    return self._buffer

  def _get_action(self, 
                  observation: Obs[np.ndarray], 
                  training: bool = True):

    env = self._env
    if self._step < self._learning_starts: # Warmup phase
      return np.array([
        env.action_space.sample() for _ in range(env.num_envs)])

    observation = self._observation_normalizer(observation, update_stats=training)
    observation = self._feature_extractor(observation)
    action = self._policy(observation)
    if training:
      action = self._action_noise(action)

    max = self._env.max_action
    return np.clip(action, -max, max)

  def get_transitions(self):

    (observation, 
     action, 
     reward, 
     next_observation, 
     done) = self._buffer.sample(self._batch_size)

    fn = self._observation_normalizer
    if fn is not None:
      observation = fn(observation, update_stats=False)
      next_observation = fn(next_observation, update_stats=False)

    fn = self._feature_extractor
    observation = fn(observation)
    next_observation = fn(next_observation)

    reward = reward.reshape(-1, 1).astype(np.float32)
    done = done.reshape(-1, 1).astype(np.float32)

    assert done.sum() == 0.0
    return (thu.torch(observation),
            thu.torch(action),
            thu.torch(reward),
            thu.torch(next_observation),
            thu.torch(done))

  def learn(self,
            training_steps: int,
            test_every_n_steps: int, 
            test_episodes: int):

    self._observation = self._env.reset()
    while self._step < training_steps:
      self.training_mode()

      action = self._get_action(self._observation)
      next_observation, reward, done, info = self._env.step(action)

      self._store_transitions(self._observation, 
                              action, 
                              reward, 
                              next_observation, 
                              done, 
                              info)
      if ((self._step > self._learning_starts) and 
          (self._step % self._train_every_n_steps < self._n_envs)):
        self.update(self._gradient_steps)

      self._observation = next_observation
      self._step += self._n_envs
      self._episode += np.sum(done)
      self._process_episode_stats(self._observation,
                                  action,
                                  reward,
                                  next_observation,
                                  done,
                                  info)

      if ((self._step > 0) and 
          (self._step % test_every_n_steps == 0)):
        self.test(test_episodes)

  def _store_transitions(self,
                         observation: Obs[np.ndarray],
                         action: np.ndarray,
                         reward: np.ndarray,
                         next_observation: Obs[np.ndarray],
                         done: np.ndarray,
                         info: List[Dict[str, Any]]):
    u"""Store transitions in the replay buffer.

    Arguments:

      observation (...): Unnormalized current observations.
      action (np.ndarray): Normalized actions.
      next_observation (...): Unnormalized next observations.
      reward (np.ndarray): Rewards.
      dones (np.ndarray): Termination or timeout signals.
      info (list): Additional information about the transitions.

    We store normalized actions and unnormalized observations.
    It also handles terminal observations, because vectorized
    environments reset automatically.
      
    """

    # As vectorized environments reset automatically, 
    # `next_observation` may be the first observation of the 
    # next episode.
    next_copy = deepcopy(next_observation)
    for i, d in enumerate(done):
      terminal = info[i].get("terminal_observation")
      if d and terminal is not None:
        if not isinstance(next_copy, dict):
          next_copy[i] = terminal
        else:
          for key in next_copy.keys():
            next_copy[key][i] = terminal[key]

    self._buffer.add(observation,
                     action,
                     reward,
                     next_copy,
                     done,
                     info)

  def _process_episode_stats(self,
                             observation: Obs[np.ndarray],
                             action: np.ndarray,
                             reward: np.ndarray,
                             next_observation: Obs[np.ndarray],
                             done: np.ndarray,
                             info: List[Dict[str, Any]]):

    for i in range(self._n_envs):
      self._episode_return[i] += reward[i]
      self._episode_step[i] += 1
      success = get_success_info(info[i])
      if success is not None:
        self._episode_success[i] = max(self._episode_success[i], success)

    if np.any(done):
      self._episode_steps.extend(self._episode_step[done])
      self._episode_successes.extend(self._episode_success[done])
      self._episode_returns.extend(self._episode_return[done])
      self._episode_success[done] = 0
      self._episode_return[done] = 0
      self._episode_step[done] = 0

    if self._step % self._log_every_n_steps < self._n_envs:
      self.logger.log("train/episode", self._episode, self._step)
      self.logger.log("train/episode_steps", np.mean(self._episode_steps), self._step)
      self.logger.log("train/success_rate", np.mean(self._episode_successes), self._step)
      self.logger.log("train/return", np.mean(self._episode_returns), self._step)

  @abstractmethod
  def update(self, gradient_steps: int):
    ...

  def test(self, n_episodes: int):

    self.evaluation_mode()
    
    episode_returns = []
    episode_successes = []

    env = self._test_env
    while len(episode_returns) < n_episodes:
      observation = env.reset()
      done = np.zeros((self._n_envs,))
      episode_success = np.zeros((self._n_envs,))
      episode_return = np.zeros((self._n_envs,))

      while not np.all(done):
        action = self._get_action(observation)
        observation, reward, done, info = env.step(action)
        for i in range(self._n_envs):
          if not done[i]:
            episode_return[i] += reward[i]
            success = get_success_info(info[i])
            if success is not None:
              episode_success[i] = max(episode_success[i], success)

      episode_returns.extend(episode_return)
      episode_successes.extend(episode_success)

    self.logger.log("test/return", 
                    np.mean(episode_returns), self._step)
    self.logger.log("test/success_rate",
                    np.mean(episode_successes), self._step)


class OnPolicyAgent(Agent):
  ...