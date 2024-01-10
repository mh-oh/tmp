
import numpy as np
import time
import pickle
import wandb

from abc import *
from collections import OrderedDict, deque
from overrides import overrides
from pathlib import Path
from typing import *

from rldev.agents.core import Node
from rldev.logging import DummyLogger
from rldev.utils.env import debug_vectorized_experience, get_success_info
from rldev.utils.time import return_elapsed_time


class Agent(metaclass=ABCMeta):

  def setup_logger(self): return DummyLogger(self)

  def __init__(self,
               config,
               env,
               test_env,
               policy):

    # this_run_path = Path(os.path.realpath(sys.argv[0]))
    # self._workspace = wdir = (Path(this_run_path.parent) 
    #                           / "data" 
    #                           / this_run_path.stem
    #                           / f"{config.run}")

    # print(f"create working directory {wdir}")
    # wdir.mkdir(parents=True, exist_ok=True)

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
  def load(self, dir: Path):

    print("loading agent...")
    # Load registered nodes.
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
               buffer,
               window=30):
    super().__init__(config, env, test_env, policy)
    self._buffer = buffer(self)

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
    self._episode_steps = deque([], maxlen=window)
    self._episode_successes = deque([], maxlen=window)
    self._episode_returns = deque([], maxlen=window)

    self._log_every_n_steps = config.log_every_n_steps

  @property
  def buffer(self):
    return self._buffer

  @overrides
  @abstractmethod
  def save(self): super().save()
  
  @overrides
  @abstractmethod
  def load(self, dir:Path): super().load(dir)

  @overrides
  def run(self, 
          epoch_steps: int, 
          test_episodes: int, *args, **kwargs):

    for epoch in range(int(self._training_steps // epoch_steps)):
      elapsed = self.train(epoch_steps)
      print(f"({epoch}) Training one epoch takes {elapsed:.2f} seconds.")
      elapsed = self.test(test_episodes)
      print(f"({epoch}) Evaluation takes {elapsed:.2f} seconds.")
      self.save()
  
  @abstractmethod
  def process_episodic_records(self, done):

    if np.any(done):
      self._episode += np.sum(done)
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
      for i in range(self._n_envs):
        success = get_success_info(experience.info[i])
        if success is not None:
          self._episode_success[i] = max(self._episode_success[i], success)
        self._episode_return[i] += experience.reward[i]
        self._episode_step[i] += 1
      self._step += self._n_envs
      self.process_episodic_records(experience.trajectory_over)

      self.process_experience(experience)
      if render:
        time.sleep(0.02)
        env.render()
      
      for _ in range(env.num_envs):
        self.env_steps += 1
        if self.env_steps % self.config.optimize_every == 0 and not dont_optimize:
          self.opt_steps += 1
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

  @return_elapsed_time
  def test(self, 
           episodes: int, 
           any_success: bool = False):

    self.evaluation_mode()
    
    episode_returns = []
    episode_successes = []

    env = self._test_env
    while len(episode_returns) < episodes:
      observation = env.reset()
      done = np.zeros((self._n_envs,))
      episode_success = np.zeros((self._n_envs,))
      episode_return = np.zeros((self._n_envs,))

      while not np.all(done):
        action = self._policy(observation)
        observation, reward, done, info = env.step(action)
        for i in range(self._n_envs):
          if not done[i]:
            episode_return[i] += reward[i]
            success = get_success_info(info[i])
            if success is not None:
              episode_success[i] = max(episode_success[i], success) if any_success else success

      episode_returns.extend(episode_return)
      episode_successes.extend(episode_success)

    self.logger.log("test/return", 
                    np.mean(episode_returns), self._step)
    self.logger.log("test/success_rate",
                    np.mean(episode_successes), self._step)


class OnPolicyAgent(Agent):
  ...