
import numpy as np

from collections import deque
from overrides import overrides
from typing import *

from rldev.environments.core import Env, DEFAULT_SIZE
from rldev.environments.maze.layout import *
from rldev.environments.maze.gym import point_maze
from rldev.environments.registry import register


def shortest_distance(layout, source, target):

  layout = np.array(layout, dtype=object)
  height, width = layout.shape
  
  def inside_maze(i, j):
    return ((i >= 0) and 
            (i < height) and 
            (j >= 0) and 
            (j < width))
  
  def neighbors(i, j):
    for u in [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)]:
      if inside_maze(*u):
        yield u  
  
  si, sj = source
  ti, tj = target
  if layout[si, sj] == 1 or layout[ti, tj] == 1:
    return None
  
  visited = np.zeros(layout.shape, dtype=bool)
  visited[si, sj] = True

  queue = deque()
  queue.append((source, 0))
  while queue:
    (i, j), distance = queue.popleft()
    if i == ti and j == tj:
      return distance
      
    for n in neighbors(i, j):
      if (layout[n] != 1 and not visited[n]):
        visited[n] = True
        queue.append((n, distance + 1))
  
  return None


class PointMaze(Env):

  _reward_modes = ["sparse",
                   "distance",
                   "path-distance-v0",
                   "path-distance-v1"]

  def __init__(self, 
               layout: List[List[Union[int, str]]],
               reward_mode: str,
               target_pvals: Sequence[float] = None,
               render_size: Tuple[int] = (DEFAULT_SIZE, DEFAULT_SIZE)):
    super().__init__()
    
    self._layout = layout
    self._reward_mode = reward_mode
    if reward_mode not in self._reward_modes:
      raise ValueError(
        f"'reward_mode' should be one of {self._reward_modes}")
    self._render_height, self._render_width = (render_size)

    cls = point_maze.PointMazeEnv
    self._env = cls(maze_map=layout,
                    render_mode=self.render_mode,
                    reward_type="sparse",
                    continuing_task=True,
                    reset_target=False,
                    target_pvals=target_pvals,
                    height=self._render_height,
                    width=self._render_width)

    self._init_object_position = None
    self._init_target_position = None

  @property
  def layout(self):
    return self._layout
  
  @property
  def maze(self):
    return self._env.maze

  @property
  @overrides
  def render_height(self):
    return self._render_height

  @property
  @overrides
  def render_width(self):
    return self._render_width

  @property
  @overrides
  def observation_space(self):
    return self._env.observation_space
  
  @property
  @overrides
  def action_space(self):
    return self._env.action_space

  @overrides
  def reset(self, *, seed=None, options=None):
    observation, info = self._env.reset(seed=seed, options=options)
    self._init_object_position = observation["achieved_goal"]
    self._init_target_position = observation["desired_goal"]
    return observation, info

  @overrides
  def step(self, action):

    (next_observation,
     reward,
     termination,
     truncation,
     info) = self._env.step(action)
    
    info["sparse_reward"] = reward
    reward = self.compute_reward(next_observation,
                                 action, next_observation, info)
    return (next_observation,
            reward,
            termination,
            truncation,
            info)

  @overrides
  def render(self):
    return self._env.render()

  @overrides
  def compute_reward(self, 
                     observation, 
                     action, 
                     next_observation, 
                     info):

    ach = next_observation["achieved_goal"]
    des = observation["desired_goal"]

    mode = self._reward_mode
    if not mode.startswith("path-distance"):
      distance = np.linalg.norm(ach - des, axis=-1)
      if mode == "distance":
        return np.exp(-distance)
      else:
        assert mode == "sparse"
        return (distance <= 0.45).astype(np.float64)

    to_index = self._env.maze.cell_xy_to_rowcol
    distance = shortest_distance(
        self._layout, to_index(ach), to_index(des))

    if mode == "path-distance-v0":
      return -distance
    elif mode == "path-distance-v1":
      return np.exp(-distance)
    else:
      raise NotImplementedError()

  @overrides
  def compute_progress(self, observation):
    
    if ((self._init_object_position is None) or
        (self._init_target_position is None)):
      raise ValueError("you must call reset() before this call")

    to_index = self._env.maze.cell_xy_to_rowcol
    def distance(x, y):
      return shortest_distance(self._layout, 
                               to_index(x), to_index(y))

    object, target = (
      observation["achieved_goal"], observation["desired_goal"])
    return (distance(object, target)
            / distance(self._init_object_position,
                       self._init_target_position))


from itertools import product
from rldev.environments.registry import register
from rldev.environments.maze.layout import registry as layouts

variants = product(layouts.items(), 
                   PointMaze._reward_modes)
for variant in variants:
  (layout_key, kwargs), reward_mode = variant
  register(f"point-maze/{layout_key}-{reward_mode}",
           f"rldev.environments.maze.point_maze:PointMaze",
           max_episode_steps=300,
           kwargs={"layout": kwargs["layout"],
                   "target_pvals": kwargs["target_pvals"],
                   "reward_mode": reward_mode})

