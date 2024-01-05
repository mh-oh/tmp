
import torch as th
from torch import nn


class Fusion(nn.Module):
  
  def __init__(self, networks):
    super().__init__()
    self.body = nn.ModuleList([fn() for fn in networks])
  
  def __getitem__(self, index):
    return self.body[index]

  @property
  def n_estimators(self):
    return len(self.body)

  def forward(self, input, reduce=True):
    output = [fn(input) for fn in self.body]
    if not reduce:
      return output
    else:
      return th.mean(th.stack(output), dim=0)

