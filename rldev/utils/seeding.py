
import numpy as np
import random

try:
  import tensorflow as tf
except:
  tf = None


def set_global_seeds(seed):
  """
    set the seed for python random, tensorflow, numpy and gym spaces

    :param seed: (int) the seed
    """
  if tf is not None:
    if hasattr(tf.random, 'set_seed'):
      tf.random.set_seed(seed)
    elif hasattr(tf.compat, 'v1'):
      tf.compat.v1.set_random_seed(seed)
    else:
      tf.set_random_seed(seed)
  np.random.seed(seed)
  random.seed(seed)
  # prng was removed in latest gym version
  if hasattr(gymnasium.spaces, 'prng'):
    gymnasium.spaces.prng.seed(seed)