"""A point mass maze environment with Gymnasium API.

The code is inspired by the D4RL repository hosted on GitHub (https://github.com/Farama-Foundation/D4RL), published in the paper
'D4RL: Datasets for Deep Data-Driven Reinforcement Learning' by Justin Fu, Aviral Kumar, Ofir Nachum, George Tucker, Sergey Levine.

Original Author of the code: Justin Fu

The modifications made involve organizing the code into different files: `maps.py`, `maze_env.py`, `point_env.py`, and `point_maze_env.py`.
As well as adding support for the Gymnasium API.

This project is covered by the Apache 2.0 License.
"""

from os import path
from typing import Dict, List, Optional, Union, Sequence

import numpy as np
from gymnasium import spaces
from gymnasium.utils.ezpickle import EzPickle

# from gymnasium_robotics.envs.point_maze.point_env import PointEnv
from gymnasium_robotics.envs.maze.maps import U_MAZE, GOAL
from gymnasium_robotics.envs.maze import maze_v4
from gymnasium_robotics.envs.maze.point import PointEnv
from gymnasium_robotics.utils.mujoco_utils import MujocoModelNames


class MazeEnv(maze_v4.MazeEnv):
  
  def __init__(self, *args, target_pvals=None, **kwargs):
    super().__init__(*args, **kwargs)

    self._target_pvals = target_pvals
    if target_pvals is not None:
      if sum(target_pvals) != 1.0:
        raise ValueError(f"probablities do not sum up to one")
      n_targets = len(self.maze.unique_goal_locations)
      n = len(target_pvals)
      if n != n_targets:
        raise ValueError(f"expected {n_targets} pvals, but got {n}")
      targets = []
      for i in range(self.maze.map_length):
        for j in range(self.maze.map_width):
          if self.maze.maze_map[i][j] == GOAL:
            targets.append((i, j))
      self._target_coords = np.array(targets, dtype=int)

  def generate_target_goal(self):

    if self._target_pvals is None:
      return super().generate_target_goal()
    else:
      drawn = self.np_random.multinomial(1, self._target_pvals)
      return self.maze.cell_rowcol_to_xy(
        self._target_coords[drawn == 1].squeeze())


class PointMazeEnv(MazeEnv, EzPickle):

  metadata = (
    {"render_modes": ["human", "rgb_array", "depth_array"],
     "render_fps": 50})

  def __init__(self,
               maze_map: List[List[Union[str, int]]] = U_MAZE,
               render_mode: Optional[str] = None,
               reward_type: str = "sparse",
               continuing_task: bool = True,
               reset_target: bool = False,
               target_pvals: Sequence[float] = None,
               **kwargs):
    point_xml_file_path = path.join(
      path.dirname(path.realpath(__file__)), "point.xml")
    super().__init__(agent_xml_path=point_xml_file_path,
                     maze_map=maze_map,
                     maze_size_scaling=1,
                     maze_height=0.4,
                     reward_type=reward_type,
                     continuing_task=continuing_task,
                     reset_target=reset_target,
                     target_pvals=target_pvals,
                     **kwargs)

    maze_length = len(maze_map)
    default_camera_config = {"distance": 12.5 
                             if maze_length > 8 else 8.8}

    self.point_env = (
      PointEnv(xml_file=self.tmp_xml_file_path,
               render_mode=render_mode,
               default_camera_config=default_camera_config,
               **kwargs))
    self._model_names = MujocoModelNames(self.point_env.model)
    self.target_site_id = self._model_names.site_name2id["target"]

    self.action_space = self.point_env.action_space
    obs_shape = self.point_env.observation_space.shape
    
    def box(shape):
      return spaces.Box(
        -np.inf, np.inf, shape=shape, dtype="float64")

    self.observation_space = (
      spaces.Dict(dict(observation=box(obs_shape),
                       achieved_goal=box((2,)),
                       desired_goal=box((2,)))))

    self.render_mode = render_mode

    EzPickle.__init__(self,
                      maze_map,
                      render_mode,
                      reward_type,
                      continuing_task,
                      reset_target,
                      **kwargs)

  def reset(self,
            *,
            seed: Optional[int] = None,
            **kwargs):
    super().reset(seed=seed, **kwargs)
    self.point_env.init_qpos[:2] = self.reset_pos

    obs, info = self.point_env.reset(seed=seed)
    obs_dict = self._get_obs(obs)
    info["success"] = bool(
      np.linalg.norm(obs_dict["achieved_goal"] - self.goal) <= 0.45)

    return obs_dict, info

  def step(self, action):
    obs, _, _, _, info = self.point_env.step(action)
    obs_dict = self._get_obs(obs)

    object = obs_dict["achieved_goal"]
    reward = self.compute_reward(object, self.goal, info)
    terminated = self.compute_terminated(object, self.goal, info)
    truncated = self.compute_truncated(object, self.goal, info)
    info["success"] = bool(
      np.linalg.norm(object - self.goal) <= 0.45)

    # Update the goal position if necessary
    self.update_goal(object)

    return obs_dict, reward, terminated, truncated, info

  def update_target_site_pos(self):
    self.point_env.model.site_pos[self.target_site_id] = (
      np.append(self.goal, 
                self.maze.maze_height 
                / 2 * self.maze.maze_size_scaling))

  def _get_obs(self, point_obs) -> Dict[str, np.ndarray]:
    achieved_goal = point_obs[:2]
    return {"observation": point_obs.copy(),
            "achieved_goal": achieved_goal.copy(),
            "desired_goal": self.goal.copy()}

  def render(self):
    return self.point_env.render()

  def close(self):
    super().close()
    self.point_env.close()

