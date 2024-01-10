
import numpy as np
import torch as th
from collections import OrderedDict

__device = "cuda"


def device():
  return th.device(__device)


def torch(x, dtype=th.float):

  if isinstance(x, (OrderedDict, dict)):
    return OrderedDict([
      (key, torch(sub, dtype=dtype)) for key, sub in x.items()])

  if isinstance(x, th.Tensor):
    return x
  if not isinstance(x, np.ndarray):
    raise ValueError(
      f"cannot convert '{type(x).__name__}' to 'th.Tensor'")
  else:
    if dtype == th.float:
      return th.from_numpy(x).float().to(device())
    elif dtype == th.long:
      return th.from_numpy(x).long().to(device())
    elif dtype == th.bool:
      return th.from_numpy(x).bool().to(device())
    else:
      raise ValueError(f"unexpected dtype='{dtype}'")


def numpy(x):
  return x.cpu().detach().numpy()