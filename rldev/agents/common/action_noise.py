
import numpy as np
import pickle

from overrides import overrides
from pathlib import Path

from rldev.agents.core import Node
from rldev.utils.random_process import *


class ContinuousActionNoise(Node):

  def __init__(self, agent, random_process_cls = GaussianProcess, *args, **kwargs):
    super().__init__(agent)
    self._random_process = random_process_cls((self._agent._env.num_envs, self._agent._env.action_dim,), *args, **kwargs)

  def __call__(self, action):
    factor = 1
    if self._agent._config.get('varied_action_noise'):
      n_envs = self._agent._env.num_envs
      factor = np.arange(0., 1. + (1./n_envs), 1./(n_envs-1)).reshape(n_envs, 1)
    
    return action + (self._random_process.sample() * self._agent._env.max_action * factor)[:len(action)]

  @overrides
  def save(self, dir: Path):
    dir.mkdir(parents=True, exist_ok=True)
    with open(dir / "_random_process.pkl", "wb") as fout:
      pickle.dump(self._random_process, fout)

  @overrides
  def load(self, dir: Path):
    with open(dir / "_random_process.pkl", "rb") as fin:
      self._random_process = pickle.load(fin)
