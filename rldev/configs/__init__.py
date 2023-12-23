
from rldev.configs.config import *
from rldev.configs.continuous_off_policy import *

from munch import AutoMunch as Conf

"""
from munch import AutoMunch, Munch, munchify
from collections.abc import *


class Ref:

  def __init__(self, key):
    if not isinstance(key, str):
      raise ValueError(
        f"expected 'str' but got '{type(key).__name__}'")
    self.key = key
    self.config = None
  
  def __repr__(self):
    return f"(lazy evaluation '{self.key}')"


class Conf(AutoMunch):

  def __setattr__(self, key, x):
    
    if (isinstance(x, Mapping) and not 
        isinstance(x, (AutoMunch, Munch, Conf))):
      x = munchify(x, Conf)
    super().__setattr__(key, x)

  def __getattr__(self, key):
    
    x = super().__getattr__(key)
    if not isinstance(x, Ref):
      return x
    
    keys = x.key.split(".")
    x = self
    print(keys)
    print(x)
    for i in range(len(keys)):
      try:
        x = x.__getattr__(keys[i])
      except AttributeError:
        raise
    return x


def items(d, prefix=[]):
  for key, x in d.items():
    if isinstance(x, dict):
      yield from items(x, prefix + [key])
    else:
      yield f"{'.'.join(prefix + [key])}", x

def get(d, key):
  keys = key.split(".")
  x = d
  for key in keys:
    x = x[key]
  return x


"""