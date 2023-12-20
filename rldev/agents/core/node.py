
from abc import *
from pathlib import Path
from typing import *


class Node(metaclass=ABCMeta):

  def __init__(self, agent):
    self._agent = agent

  @property
  def agent(self):
    return self._agent

  @abstractmethod
  def save(self, dir: Path):
    pass

  @abstractmethod
  def load(self, dir: Path):
    pass

