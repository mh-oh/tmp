
from abc import *
from gymnasium import spaces
from typing import *

from torch import nn

from rldev.utils.env import flatten_observation, flatten_space
from rldev.utils.structure import recursive_getitem


class Extractor(nn.Module):

  def __init__(self, 
               observation_space: spaces.Space):
    super().__init__()
    self._observation_space = observation_space

  @property
  def observation_space(self):
    return self._observation_space

  @property
  def feature_space(self):
    ...


class Combine(Extractor):
  
  def __init__(self, 
               observation_space: spaces.Dict,
               keys: Optional[Sequence[str]] = None):

    if keys is None:
      super().__init__(observation_space)
    else:
      # Dict observation space with specified keys.
      new_observation_space = spaces.Dict()
      def set(space, key, x):
        *keys, key = key
        for k in keys:
          if k not in space.spaces:
            space[k] = spaces.Dict()
          space = space[k]
        space[key] = x

      def fn(key):
        return (key,) if isinstance(key, str) else key

      for key in map(fn, keys):
        box = recursive_getitem(observation_space, key)
        if not isinstance(box, spaces.Box):
          raise ValueError(f"Box space is expected at '{key}'")
        set(new_observation_space, key, box)
      super().__init__(new_observation_space)

  def forward(self, 
              observation: Dict[str, Any]):
    return flatten_observation(self.observation_space,
                               observation)

  @property
  def feature_space(self):
    return flatten_space(self.observation_space)

