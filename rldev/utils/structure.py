
import numpy as np
import torch as th

from collections import OrderedDict
from collections.abc import *
from itertools import tee


def dataclass(cls, /, **kwargs):

  from dataclasses import dataclass, fields
  class C(dataclass(cls, **kwargs)):
    __qualname__ = cls.__qualname__
    def __iter__(self):
      for field in fields(self):
        yield getattr(self, field.name)
  return C


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


def instanceof(*inputs, type):
  return all(isinstance(x, type) for x in inputs)


def stack(inputs, axis):

  if instanceof(*inputs, type=np.ndarray):
    return np.stack(inputs, axis=axis)
  if instanceof(*inputs, type=th.Tensor):
    return th.stack(inputs, dim=axis)

  if instanceof(*inputs, type=OrderedDict):
    output = OrderedDict()
    for dict in inputs:
      for key, x in dict.items():
        output.setdefault(key, []).append(x)
    for key in output:
      output[key] = stack(output[key], axis)
    return output


def concatenate(inputs, axis):

  if instanceof(*inputs, type=np.ndarray):
    return np.concatenate(inputs, axis=axis)
  if instanceof(*inputs, type=th.Tensor):
    return th.cat(inputs, dim=axis)

  if instanceof(*inputs, type=OrderedDict):
    output = OrderedDict()
    for dict in inputs:
      for key, x in dict.items():
        output.setdefault(key, []).append(x)
    for key in output:
      output[key] = concatenate(output[key], axis)
    return output


def recursive_items(d):
  u"""Nested iteration over `(key, value)` pairs.
  `key` is a tuple of nested keys in the dictionary `d`."""

  def f(d, parents):
    for key, x in d.items():
      if not isinstance(x, (dict, OrderedDict)):
        yield (*parents, key), x
      else:
        yield from f(x, (*parents, key))

  yield from f(d, ())


def recursive_keys(d):
  u"""Nested iteration over `key`s.
  `key` is a tuple of nested keys in the dictionary `d`."""

  for key, _ in recursive_items(d):
    yield key


def recursive_setitem(d, key, x):
  *keys, key = key
  for k in keys:
    if k not in d:
      d[k] = type(d)()
    d = d[k]
  d[key] = x


def recursive_getitem(d, key):
  x = d
  for k in key:
    x = x[k]
  return x

