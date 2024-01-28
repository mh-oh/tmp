
import csv
import json
import numpy as np
import os
import sys
import time
import wandb

from collections import defaultdict
from overrides import overrides
from pathlib import Path
from tabulate import tabulate


class DummyLogger:

  def define(self, *metrics, step_metric="train/step"):
    ...

  def log(self, key, x, step):
    ...


class WandbLogger(DummyLogger):

  def __init__(self):
    super().__init__()
    self._metrics = {}

  def define(self, *metrics, step_metric="train/step"):

    for key in metrics:
      self._metrics[key] = step_metric
      wandb.define_metric(
        key, step_metric=step_metric)

  def log(self, key, x, step):
    metrics = self._metrics
    if key not in metrics:
      raise ValueError(f"unknown metric '{key}'")
    wandb.log({key: x, metrics[key]: step})

