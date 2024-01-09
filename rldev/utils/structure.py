
import numpy as np

from collections import defaultdict, OrderedDict
from collections.abc import *
from itertools import tee


class AttrDict(dict):
  """
    Behaves like a dictionary but additionally has attribute-style access
    for both read and write.
    e.g. x["key"] and x.key are the same,
    e.g. can iterate using:  for k, v in x.items().
    Can sublcass for specific data classes; must call AttrDict's __init__().
    """
  def __init__(self, *args, **kwargs):
    dict.__init__(self, *args, **kwargs)
    self.__dict__ = self

  def copy(self):
    """
        Provides a "deep" copy of all unbroken chains of types AttrDict, but
        shallow copies otherwise, (e.g. numpy arrays are NOT copied).
        """
    return type(self)(**{k: v.copy() if isinstance(v, AttrDict) else v for k, v in self.items()})


class AnnotatedAttrDict(AttrDict):
  """
  This is an AttrDict that accepts tuples of length 2 as values, where the
  second element is an annotation.
  """
  def __init__(self, *args, **kwargs):
    argdict = dict(*args, **kwargs)
    valuedict = {}
    annotationdict = {}
    for k, va in argdict.items():
      if hasattr(va, '__len__') and len(va) == 2 and type(va[1]) == str:
        v, a = va
        valuedict[k] = v
        annotationdict[k] = a
      else:
        valuedict[k] = va
    super().__init__(self, **valuedict)


def zip_items(*dicts):
  d, *dicts = dicts
  for key, x in d.items():
    yield key, (x, *map(lambda d: d[key], dicts))


def recursive_map(fn, *args):
  if not all(isinstance(a, Mapping) for a in args):
    return fn(*args)
  else:
    return type(args[0])(
      (key, recursive_map(fn, *x)) for key, x in zip_items(*args))


def recursive_get(index, *args, copy=False):
  def fn(x):
    return x[index].copy() if copy else x[index]
  return recursive_map(fn, *args)


def copy(x):
  if isinstance(x, np.ndarray):
    return np.copy(x)
  elif isinstance(x, Mapping):
    def fn(x):
      if not isinstance(x, np.ndarray):
        raise AssertionError(f"{x}: {type(x).__name__}")
      return np.copy(x)
    return recursive_map(fn, x)
  else:
    raise AssertionError(f"{x}: {type(x).__name__}")


def resolve_ellipsis(index, ndims):
  
  if not isinstance(index, tuple):
    raise ValueError()
  try:
    i = index.index(...)
  except:
    return index
  else:
    return (*index[:i], *((slice(None, None, None),) * (ndims - len(index) + 1)), *index[i+1:])


def nest(items, sep=".", cls=dict):

  if not isinstance(items, Iterable):
    raise ValueError(f"")

  dict = cls()
  for key, x in items:
    keys = key.split('.')
    curr = dict
    for k in keys[:-1]:
      curr = curr.setdefault(k, cls())
    curr[keys[-1]] = x
  return dict


class ArrDict:

  def __init__(self, data, shape):
    
    self._data = data
    self._shape = shape
    
    def check(data):
      for key, x in data.items():
        if not isinstance(key, str):
          raise KeyError(
            f"for key='{key}', "
            f"expect 'str' but got '{type(key).__name__}'")
        if isinstance(x, Mapping):
          check(x)
        else:
          if not isinstance(x, np.ndarray):
            raise ValueError(
              f"for key='{key}', "
              f"expect 'np.ndarray' but got '{type(x).__name__}'")
          if x.shape[:len(shape)] != shape:
            raise ValueError()

    check(data)
  
  def __repr__(self):
    return self._data.__repr__()

  @property
  def shape(self):
    return self._shape

  def items(self, sep="."):

    def fn(data, prefix, sep):
      for key, x in data.items():
        key = sep.join((*prefix, key))
        if not isinstance(x, Mapping):
          yield key, x
        else:
          yield from fn(x, key, sep)

    yield from fn(self._data, (), sep)

  def keys(self, sep="."):
    for key, x in self.items(sep):
      yield key
  
  def values(self):
    for key, x in self.items():
      yield x

  def get(self, key, sep="."):

    x = self._data
    try:
      for k in key.split(sep):
        x = x[k]
    except KeyError:
      raise KeyError(f"'{key}'") from None
    return x

  def set(self, key, data, sep="."):

    if not isinstance(data, np.ndarray):
      raise ValueError(f"")

    *keys, last = key.split(sep)
    try:
      x = self._data
      for k in keys:
        x = x[k]
    except KeyError:
      raise KeyError(f"'{key}'") from None
    else:
      try:
        before = x[last]
        if before.shape != data.shape:
          raise ValueError(f"")
        x[last] = data
      except KeyError:
        raise KeyError(f"{key}") from None

  def copy(self):
    return ArrDict(
      recursive_map(lambda x: np.copy(x), self._data), shape=self.shape)

  def _resolve(self, index):

    if not isinstance(index, tuple):
      index = (index,)
    if len(index) > len(self._shape):
      raise ValueError(f"too many indices")

    ndims = len(self._shape)
    return (resolve_ellipsis(index, ndims=ndims) 
            + (slice(None, None, None),) * (ndims - len(index)))

  def __getitem__(self, index):
    
    shapes = []
    def get(x):
      y = x[self._resolve(index) + (...,)]
      shapes.append(y.shape[:-(len(self.shape) - len(x.shape))])
      return y
    
    data = recursive_map(get, self._data)
    assert len(set(shapes)) == 1
    
    return ArrDict(data, shape=shapes[0])

  def __setitem__(self, index, data):

    def set(x, y):
      x[self._resolve(index) + (...,)] = y

    if isinstance(data, Mapping):
      recursive_map(set, self._data, data)
    elif isinstance(data, ArrDict):
      raise NotImplementedError("which rules should we follow?")


def batchify(length, size):
  for i in range(0, length, size):
    yield i, min(i + size, length)


def chunk(sequence, n):
  
  def _do(sequence, n):
    k, m = divmod(len(sequence), n)
    for i in range(n):
      yield sequence[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)]

  if not isinstance(sequence, int):
    yield from _do(sequence, n)
  else:
    for sub in _do(list(range(sequence)), n):
      yield len(sub)

def isiterable(obj):
  try:
    iter(obj)
  except:
    return False
  else:
    return True


def pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)