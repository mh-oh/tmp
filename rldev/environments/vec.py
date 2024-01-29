
import gymnasium as gym
import numpy as np

from stable_baselines3.common.vec_env import DummyVecEnv


def make_env(name: str, seed: int, rank: int = 0):
  u"""Creates an environment.

  Arguments:
    name (str): Identifier string of an environment.
    rank (int): Rank of the subprocess.
    seed (int): Inital seed for RNG.
  
  Returns:
    Callable: ...
  """

  def thunk():
    env = gym.make(name, render_mode="rgb_array")
    env.reset(seed=seed + rank)
    return env

  return thunk


def make_vec_env(name: str, seed: int, n_envs: int):
  u"""Creates vectorized environments.

  Arguments:
    name (str): An identifier string of environments.
    seed (int): Inital seed for RNG.
    n_envs (int): Number of environments.
  """
  venv = DummyVecEnv(
    [make_env(name, seed, i) for i in range(n_envs)])
  
  venv.action_dim = venv.action_space.shape[0]
  venv.max_action = venv.action_space.high[0]


  venv.goal_env = False
  venv.goal_dim = 0

  venv.goal_env = True
  venv.compute_reward = venv.envs[0].compute_reward
  for key in ["desired_goal"]:
    venv.goal_dim += int(np.prod(venv.observation_space[key].shape))
  state_dim = 0
  for key in ["observation"]:
    state_dim += int(np.prod(venv.observation_space[key].shape))
  venv.state_dim = state_dim

  return venv