
from copy import deepcopy
from gymnasium import spaces
from typing import Tuple, Dict, Union, Any

import numpy as np


class RunningMeanStddev:

  def __init__(self, 
               shape: Tuple[int, ...] = (),
               epsilon: float = 1e-4):
    """Running mean and stddev of a data stream.
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

      shape (tuple of ints): The shape of the data stream's items.
      epsilon (float): Helps with arithmetic issues
    """

    self._mean = np.zeros(shape, np.float64)
    self._var = np.ones(shape, np.float64)
    self._count = epsilon

  @property
  def mean(self):
    return self._mean
  
  @property
  def var(self):
    return self._var

  def update(self, x: np.ndarray):

    batch_mean = np.mean(x, axis=0)
    batch_var = np.var(x, axis=0)
    batch_count = x.shape[0]
    self.update_from_moments(batch_mean, batch_var, batch_count)

  def update_from_moments(self, 
                          batch_mean: np.ndarray, 
                          batch_var: np.ndarray, 
                          batch_count: float):

    delta = batch_mean - self._mean
    tot_count = self._count + batch_count

    new_mean = self._mean + delta * batch_count / tot_count
    m_a = self._var * self._count
    m_b = batch_var * batch_count
    m_2 = m_a + m_b + np.square(delta) * self._count * batch_count / (self._count + batch_count)
    new_var = m_2 / (self._count + batch_count)

    new_count = batch_count + self._count

    self._mean = new_mean
    self._var = new_var
    self._count = new_count


class ObservationNormalizer:

  def __init__(self,
               observation_space: spaces.Space):
    self._observation_space = observation_space

  def __call__(self, 
               observation: Union[np.ndarray, Dict[str, np.ndarray]],
               **kwargs: Dict[str, Any]):
    raise NotImplementedError()


class MeanStdNormalizer(ObservationNormalizer):

  def __init__(self,
               observation_space: spaces.Space,
               clip: float = 5.0,
               epsilon: float = 1e-8):
    super().__init__(observation_space)

    self._clip = clip
    self._epsilon = epsilon

    space = deepcopy(observation_space)
    if isinstance(space, spaces.Dict):
      self._rms = {
        key: RunningMeanStddev(shape=subspace.shape) 
          for key, subspace in space.spaces.items()}
      # Update observation space when using image
      # See explanation below and GH #1214
      """
      for key in self._rms.keys():
        if is_image_space(space.spaces[key]):
          space.spaces[key] = (
            spaces.Box(-clip, 
                       +clip,
                       shape=self.obs_spaces[key].shape,
                       dtype=np.float32))
      """
    else:
      self._rms = RunningMeanStddev(shape=space.shape)
      # Update observation space when using image
      # See GH #1214
      # This is to raise proper error when
      # VecNormalize is used with an image-like input and
      # normalize_images=True.
      # For correctness, we should also update the bounds
      # in other cases but this will cause backward-incompatible change
      # and break already saved policies.
      """
      if is_image_space(space):
        space = (
          spaces.Box(-clip, 
                     +clip,
                     shape=self.observation_space.shape,
                     dtype=np.float32))
      """

  def __call__(self, 
               observation: Union[np.ndarray, Dict[str, np.ndarray]],
               update_stats: bool = True):

    if update_stats:
      if isinstance(observation, dict):
        for key in self._rms.keys():
          self._rms[key].update(observation[key])
      else:
        self._rms.update(observation)

    def _transform(x, rms):
      mean, var = rms.mean, rms.var
      return np.clip((x - mean) / np.sqrt(var + self._epsilon), 
                     -self._clip, self._clip)

    obs = deepcopy(observation)
    if isinstance(observation, dict):
      assert isinstance(self._rms, dict)
      for key in self._rms:
        obs[key] = _transform(
          observation[key], self._rms[key])
    else:
      obs = _transform(observation, self._rms)
    return obs

