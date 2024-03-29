
import numpy as np
import pickle


def save(path, x):
  with open(path, "wb") as fout:
    pickle.dump(x, fout)


def save_np(path, x):
  with open(path, "wb") as fout:
    np.save(fout, x)


def load(obj, path, x):
  with open(path, "rb") as fin:
    setattr(obj, x, pickle.load(fin))


def check(obj, path, x):
  with open(path, "rb") as fin:
    y = pickle.load(fin)
  x = getattr(obj, x)
  if x != y:
    raise ValueError(f"expected {x} but got {y}")