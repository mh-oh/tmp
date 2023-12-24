
import gym
import gym.spaces as spaces


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