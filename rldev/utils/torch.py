
import torch as th


__device = "cuda"


def device():
  return th.device(__device)


def torch(x, type=th.float):
  if isinstance(x, th.Tensor): return x
  elif type == th.float:
    return th.FloatTensor(x).to(device())
  elif type == th.long:
    return th.LongTensor(x).to(device())
  elif type == th.bool:
    return th.BoolTensor(x).to(device())


def numpy(x):
  return x.cpu().detach().numpy()