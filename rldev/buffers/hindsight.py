
import numpy as np
import torch as th

from gymnasium import spaces
from typing import Dict, Any, Callable, Union

from rldev.buffers.basic import EpisodicDictBuffer, DictExperience
from rldev.utils.structure import recursive_map


DictObs = Dict[str, Union[np.ndarray, "DictObs"]]
BoxObs = np.ndarray
Obs = Union[DictObs, BoxObs]
RFunction = Callable[[Obs, np.ndarray, Obs], np.ndarray]


class HindsightBuffer(EpisodicDictBuffer):

  def __init__(self, 
               n_envs: int, 
               capacity: int, 
               observation_space: Union[spaces.Dict, spaces.Box], 
               action_space: spaces.Box,
               compute_reward: RFunction,
               mode: str,
               handle_timeouts: bool = True):
    super().__init__(n_envs, 
                     capacity, 
                     observation_space, 
                     action_space,
                     handle_timeouts)

    capacity, n_envs = self._capacity, self._n_envs
    self._episode_cursor = np.zeros((n_envs,), dtype=int)
    self._episode_length = np.zeros((capacity, n_envs), dtype=int)
    self._episode_starts = np.zeros((capacity, n_envs), dtype=int)

    self._compute_reward = compute_reward
    self._fut, self._act, self._ach, self._beh = parse_hindsight_mode(mode)

  def none(self, index):

    index = np.unravel_index(index, self._episode_length.shape)
    return self.get(index)

  def real(self, index):

    (observations, 
     actions, 
     rewards, 
     next_observations, 
     dones) = self.none(index)
    
    new_goals = np.copy(observations.get("desired_goal")) #############

    observations["desired_goal"] = new_goals
    next_observations["desired_goal"] = new_goals
    
    return DictExperience(observations,
                          actions,
                          rewards,
                          next_observations,
                          dones)

  def future(self, index):

    (observations, 
     actions, 
     rewards, 
     next_observations, 
     dones) = self.none(index)

    index = np.unravel_index(index, self._episode_length.shape)
    batch_index, env_index = index

    episode_starts = self._episode_starts[index]
    episode_length = self._episode_length[index]
    future_index = np.random.randint(batch_index, episode_starts + episode_length) % self._capacity
    new_goals = self._next_observations["achieved_goal"][future_index, env_index]

    observations["desired_goal"] = new_goals
    next_observations["desired_goal"] = new_goals
    
    return DictExperience(observations,
                          actions,
                          rewards,
                          next_observations,
                          dones)
  
  def desired(self, index):
    
    (observations, 
     actions, 
     rewards, 
     next_observations, 
     dones) = self.none(index)
    
    batch_index = np.random.choice(self._cursor, size=index.shape)
    env_index = np.random.choice(self._n_envs, size=index.shape)
    new_goals = self._observations["desired_goal"][batch_index, env_index]

    observations["desired_goal"] = new_goals
    next_observations["desired_goal"] = new_goals

    return DictExperience(observations,
                          actions,
                          rewards,
                          next_observations,
                          dones)

  def achieved(self, index):
    
    (observations, 
     actions, 
     rewards, 
     next_observations, 
     dones) = self.none(index)
    
    batch_index = np.random.choice(self._cursor, size=index.shape)
    env_index = np.random.choice(self._n_envs, size=index.shape)
    new_goals = self._next_observations["achieved_goal"][batch_index, env_index]

    observations["desired_goal"] = new_goals
    next_observations["desired_goal"] = new_goals

    return DictExperience(observations,
                          actions,
                          rewards,
                          next_observations,
                          dones)

  def behavior(self, index):
    
    (observations, 
     actions, 
     rewards, 
     next_observations, 
     dones) = self.none(index)
    
    batch_index = np.random.choice(self._cursor, size=index.shape)
    env_index = np.random.choice(self._n_envs, size=index.shape)
    new_goals = self._observations["desired_goal"][batch_index, env_index] ##################

    observations["desired_goal"] = new_goals
    next_observations["desired_goal"] = new_goals

    return DictExperience(observations,
                          actions,
                          rewards,
                          next_observations,
                          dones)

  def sample(self, size: int):

    is_done_episode = self._episode_length > 0
    if not np.any(is_done_episode):
      raise ValueError(f"No episode ends")

    episode_index = np.flatnonzero(is_done_episode)
    index = np.random.choice(episode_index, size=size, replace=True)

    if len(self) > 25000:
      fut_batch_size, act_batch_size, ach_batch_size, beh_batch_size, real_batch_size = np.random.multinomial(
          size, [self._fut, self._act, self._ach, self._beh, 1.])
    else:
      fut_batch_size, act_batch_size, ach_batch_size, beh_batch_size, real_batch_size = size, 0, 0, 0, 0
    
    future, actual, achieved, behavior, real = np.array_split(index, 
      np.cumsum([fut_batch_size, act_batch_size, ach_batch_size, beh_batch_size]))

    real = self.real((real))
    future = self.future((future))
    actual = self.desired((actual))
    achieved = self.achieved((achieved))
    behavior = self.behavior((behavior))

    def concatenate(*args):
      return np.concatenate(args, axis=0)

    observations = recursive_map(
      concatenate, *(real.observation, 
                     future.observation, 
                     actual.observation, 
                     achieved.observation, 
                     behavior.observation))
    next_observations = recursive_map(
      concatenate, *(real.next_observation, 
                     future.next_observation, 
                     actual.next_observation, 
                     achieved.next_observation, 
                     behavior.next_observation))
    actions = concatenate(real.action,
                          future.action,
                          actual.action,
                          achieved.action,
                          behavior.action)
    dones = concatenate(real.done,
                        future.done,
                        actual.done,
                        achieved.done,
                        behavior.done)

    rewards = self._compute_reward(observations, actions,
                                   next_observations)

    return (observations,
            actions,
            rewards,
            next_observations,
            dones)


def parse_hindsight_mode(hindsight_mode : str):
  if 'future_' in hindsight_mode:
    _, fut = hindsight_mode.split('_')
    fut = float(fut) / (1. + float(fut))
    act = 0.
    ach = 0.
    beh = 0.
  elif 'futureactual_' in hindsight_mode:
    _, fut, act = hindsight_mode.split('_')
    non_hindsight_frac = 1. / (1. + float(fut) + float(act))
    fut = float(fut) * non_hindsight_frac
    act = float(act) * non_hindsight_frac
    ach = 0.
    beh = 0.
  elif 'futureachieved_' in hindsight_mode:
    _, fut, ach = hindsight_mode.split('_')
    non_hindsight_frac = 1. / (1. + float(fut) + float(ach))
    fut = float(fut) * non_hindsight_frac
    act = 0.
    ach = float(ach) * non_hindsight_frac
    beh = 0.
  elif 'rfaa_' in hindsight_mode:
    _, real, fut, act, ach = hindsight_mode.split('_')
    denom = (float(real) + float(fut) + float(act) + float(ach))
    fut = float(fut) / denom
    act = float(act) / denom
    ach = float(ach) / denom
    beh = 0.
  elif 'rfaab_' in hindsight_mode:
    _, real, fut, act, ach, beh = hindsight_mode.split('_')
    denom = (float(real) + float(fut) + float(act) + float(ach) + float(beh))
    fut = float(fut) / denom
    act = float(act) / denom
    ach = float(ach) / denom
    beh = float(beh) / denom
  else:
    fut = 0.
    act = 0.
    ach = 0.
    beh = 0.

  return fut, act, ach, beh
