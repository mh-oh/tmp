
import numpy as np
import pickle

from overrides import overrides
from pathlib import Path
from rldev.agents.core import Node


class RunningMeanStd(object):
  # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
  def __init__(self, epsilon=1e-4, shape=()):
    self.mean = np.zeros(shape, 'float64')
    self.var = np.ones(shape, 'float64')
    self.count = epsilon

  def update(self, x):
    batch_mean = np.mean(x, axis=0, keepdims=True)
    batch_var = np.var(x, axis=0, keepdims=True)
    batch_count = x.shape[0]
    self.update_from_moments(batch_mean, batch_var, batch_count)

  def update_from_moments(self, batch_mean, batch_var, batch_count):
    delta = batch_mean - self.mean
    tot_count = self.count + batch_count

    new_mean = self.mean + delta * batch_count / tot_count
    m_a = self.var * (self.count)
    m_b = batch_var * (batch_count)
    M2 = m_a + m_b + np.square(delta) * self.count * batch_count / (tot_count)
    new_var = M2 / (tot_count)

    self.mean = new_mean
    self.var = new_var
    self.count = tot_count


class Normalizer(Node):

  def __init__(self, agent, normalizer):
    super().__init__(agent)
    self._normalizer = normalizer
  
  def __call__(self, *args, **kwargs):
    if self._agent.training:
      self._normalizer.read_only = False
    else:
      self._normalizer.read_only = True
    return self._normalizer(*args, **kwargs)

  @overrides
  def save(self, dir: Path):
    dir.mkdir(parents=True, exist_ok=True)
    with open(dir / "_normalizer.pkl", "wb") as fout:
      pickle.dump(self._normalizer, fout)

  @overrides
  def load(self, dir: Path):
    with open(dir / "_normalizer.pkl", "rb") as fin:
      self._normalizer = pickle.load(fin)


# Below from https://github.com/ShangtongZhang/DeepRL/blob/master/deep_rl/utils/normalizer.py


class BaseNormalizer:
  def __init__(self, read_only=False):
    self.read_only = read_only

  def set_read_only(self):
    self.read_only = True

  def unset_read_only(self):
    self.read_only = False

  def state_dict(self):
    return None

  def load_state_dict(self, _):
    return


class MeanStdNormalizer(BaseNormalizer):
  def __init__(self, read_only=False, clip_before=200.0, clip_after=5.0, epsilon=1e-8):
    BaseNormalizer.__init__(self, read_only)
    self.read_only = read_only
    self.rms = None
    self.clip_before = clip_before
    self.clip_after = clip_after
    self.epsilon = epsilon

  def __call__(self, x, update=True):
    x = np.clip(np.asarray(x), -self.clip_before, self.clip_before)
    if self.rms is None:
      self.rms = RunningMeanStd(shape=(1, ) + x.shape[1:])
    if not self.read_only and update:
      self.rms.update(x)
    return np.clip((x - self.rms.mean) / np.sqrt(self.rms.var + self.epsilon), -self.clip_after, self.clip_after)

  def state_dict(self):
    if self.rms is not None:
      return {'mean': self.rms.mean, 'var': self.rms.var, 'count': self.rms.count}

  def load_state_dict(self, saved):
    self.rms.mean = saved['mean']
    self.rms.var = saved['var']
    self.rms.count = saved['count']


class RescaleNormalizer(BaseNormalizer):
  def __init__(self, coef=1.0):
    BaseNormalizer.__init__(self)
    self.coef = coef

  def __call__(self, x, *unused_args):
    if not isinstance(x, torch.Tensor):
      x = np.asarray(x)
    return self.coef * x


class ImageNormalizer(RescaleNormalizer):
  def __init__(self):
    RescaleNormalizer.__init__(self, 1.0 / 255)


class SignNormalizer(BaseNormalizer):
  def __call__(self, x, *unused_args):
    return np.sign(x)
