
from abc import *
from pathlib import Path
from typing import *


class Node(metaclass=ABCMeta):

  def __init__(self, agent, 
               disable_save=False, disable_load=False):
    self._agent = agent
    if disable_save:
      self.disable_save()
    if disable_load:
      self.disable_load()

  @property
  def agent(self):
    return self._agent

  @abstractmethod
  def save(self, dir: Path):
    pass

  @abstractmethod
  def load(self, dir: Path):
    pass

  def disable_load(self):
    self.load = lambda dir: None

  def disable_save(self):
    self.save = lambda dir: None
