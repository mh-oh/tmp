
import torch as th
from torch import nn


class EnsembleReward(nn.Module):
  
  def __init__(self, networks):
    super().__init__()
    self.body = nn.ModuleList([thunk() for thunk in networks])
  
  def __getitem__(self, index):
    return self.body[index]
  
  def forward(self, input, reduce="mean"):
    return th.mean(
      th.stack([fn(input) for fn in self.body]), dim=0)

