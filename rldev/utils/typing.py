
import numpy as np
import torch as th

from typing import *


T = TypeVar("T", np.ndarray, th.Tensor)
BoxObs = T
DictObs = Dict[str, Union[T, "DictObs"]]
Obs = Union[BoxObs, DictObs]

