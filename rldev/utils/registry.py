
from torch import nn


_registry = (
  {"leaky-relu": nn.LeakyReLU,
   "tanh": nn.Tanh,
   "sigmoid": nn.Sigmoid,
   "relu": nn.ReLU,
   "identity": nn.Identity})


def get(name):
  return _registry[name]