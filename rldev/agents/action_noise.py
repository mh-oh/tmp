
from abc import ABC, abstractmethod
import numpy as np
from gymnasium import spaces


class ActionNoise(ABC):

  def reset(self):
    """
    Call end of episode reset for the noise
    """
    pass

  @abstractmethod
  def __call__(self, action: np.ndarray):
    raise NotImplementedError()


class GaussianActionNoise(ActionNoise):

  def __init__(self,
               mean: float, 
               stddev: float):
    super().__init__()

    self._mu = mean
    self._sigma = stddev

  def __call__(self, action: np.ndarray):
    n_envs, d = action.shape
    return action + (np.random.randn(n_envs, d) * self._sigma + self._mu)

