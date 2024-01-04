
import numpy as np

from collections import deque
from overrides import overrides

from gymnasium_robotics.envs.maze import point_maze

from rldev.environments.core import Env
from rldev.environments.maze.layout import *
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


class PointMazeV0(Env):

  def __init__(self, 
               layout,
               reward_mode):
    super().__init__()
    
    self._layout = layout
    self._reward_mode = reward_mode

    cls = point_maze.PointMazeEnv
    self._env = cls(maze_map=layout,
                    render_mode=self.render_mode,
                    reward_type=reward_mode,
                    continuing_task=True,
                    reset_target=False)

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
    return self._env.reset(seed=seed, options=options)

  @overrides
  def step(self, action):

    (observation,
     reward,
     termination,
     truncation,
     info) = self._env.step(action)
    
    r = self.compute_teacher_reward
    info["teacher_reward"] = r(observation,
                               action, observation, info)
    return (observation,
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
    r = self._env.compute_reward
    return r(next_observation["achieved_goal"], 
             observation["desired_goal"], 
             info)

  @overrides
  def compute_teacher_reward(self, 
                             observation, 
                             action, 
                             next_observation, 
                             info):
    return self.compute_reward(observation,
                               action,
                               next_observation,
                               info)


class PointMazeV1(PointMazeV0):
  
  @overrides
  def compute_teacher_reward(self, 
                             observation, 
                             action, 
                             next_observation, 
                             info):

    object = next_observation["achieved_goal"]
    target = observation["desired_goal"]
    
    to_index = self._env.maze.cell_xy_to_rowcol
    distance = shortest_distance(
      self._layout, to_index(object), to_index(target))
    if distance is None:
      raise AssertionError()

    return -distance


class PointMazeV2(PointMazeV1):

  @overrides
  def compute_teacher_reward(self, 
                             observation, 
                             action, 
                             next_observation, 
                             info):

    object = next_observation["achieved_goal"]
    target = observation["desired_goal"]
    
    to_index = self._env.maze.cell_xy_to_rowcol
    distance = shortest_distance(
      self._layout, to_index(object), to_index(target))
    if distance is None:
      raise AssertionError()

    return np.exp(-distance)


def path(cls):
  return f"rldev.environments.maze.point_maze:{cls}"
envs = [
  ("point-maze-v0", path("PointMazeV0")),
  ("point-maze-v1", path("PointMazeV1")),
  ("point-maze-v2", path("PointMazeV2"))]


from itertools import product
from gymnasium.envs.registration import register
from rldev.environments.maze.layout import registry as layouts

variants = product(envs, 
                   layouts.items(), 
                   ["sparse", "dense"])

for variant in variants:
  ((env_key, env_path), 
   (layout_key, layout), reward_mode) = variant

  id = f"{env_key}/{layout_key}-{reward_mode}"
  register(id,
           env_path,
           max_episode_steps=300,
           kwargs={"layout": layout,
                   "reward_mode": reward_mode})

