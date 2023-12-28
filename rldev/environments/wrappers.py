
import gym
import gym.spaces as spaces
import gymnasium

from gymnasium.utils.step_api_compatibility import convert_to_done_step_api


class BoxObservation(gym.wrappers.FlattenObservation):

  def __init__(self, env):
    u""""""

    observation_space = env.observation_space
    if not isinstance(observation_space, spaces.Dict):
      raise ValueError(f"")
    self.dict_observation_space = observation_space

    super().__init__(env)

  def dict_observation(self, observation):
    return spaces.unflatten(
      self.dict_observation_space, observation)


class SuccessInfo(gym.Wrapper):

  def __init__(self, env):
    super().__init__(env)

  def step(self, action):
    *step, info = super().step(action)
    if "is_success" in info:
      info["success"] = info["is_success"]
    return *step, info


class GymApi(gymnasium.Wrapper):

  def __init__(self, env):
    super().__init__(env)
  
  def reset(self):
    observation, info = super().reset()
    return observation
  
  def step(self, action):
    step_return = super().step(action)
    return convert_to_done_step_api(step_return)

  def seed(self, seed=None):
    ...