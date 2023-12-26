"""
This is old replay buffer. Does not work with HER. 
Sole advantage is that it does not wait  for the trajectory to finish to add things to the buffer,
  which is important for long Mujoco tasks. 
"""

import gym
import numpy as np
import os
import pickle

from rldev.agents.core import Node
from rldev.buffers.core.buffer import ReplayBuffer as Buffer


class OldReplayBuffer(Node):

  def __init__(self, agent):
    super().__init__(agent)

    self.size = None
    self.goal_space = None
    self.hindsight_buffer = None
    self.buffer = None
    self.save_buffer = None

    self.size = self._agent.config.replay_size

    env = self._agent.env
    if type(env.observation_space) == gym.spaces.Dict:
      observation_space = env.observation_space.spaces["observation"]
      self.goal_space = env.observation_space.spaces["desired_goal"]
      raise NotImplementedError("This buffer no longer supports goal spaces; use OnlineHERBuffer")
    else:
      observation_space = env.observation_space

    items = [("state", observation_space.shape),
             ("action", env.action_space.shape), ("reward", (1,)),
             ("next_state", observation_space.shape), ("done", (1,))]

    self.buffer = Buffer(self.size, items)

  def _process_experience(self, experience):
    if getattr(self._agent, 'logger'):
      self._agent.logger.add_tabular('Replay buffer size', len(self.buffer))
    done = np.expand_dims(experience.done, 1) # format for replay buffer
    reward = np.expand_dims(experience.reward, 1) # format for replay buffer
    action = experience.action

    state = experience.state
    next_state = experience.next_state
    self.buffer.add_batch(state, action, reward, next_state, done)
  
  def sample(self, batch_size, to_torch=True):
    states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)
    gammas = self._agent.config.gamma * (1-dones)
    
    if self._agent._observation_normalizer is not None:
      states = self._agent._observation_normalizer(states, update=False).astype(np.float32)
      next_states = self._agent._observation_normalizer(next_states, update=False).astype(np.float32)

    if to_torch:
      return (self._agent.torch(states), self._agent.torch(actions),
            self._agent.torch(rewards), self._agent.torch(next_states),
            self._agent.torch(gammas))
    else:
      return (states, actions, rewards, next_states, gammas)
        
  def __len__(self):
    return len(self.buffer)

  def save(self, save_folder):
    if self._agent.config.save_replay_buf or self.save_buffer:
      state = self.buffer._get_state()
      with open(os.path.join(save_folder, "{}.pickle".format(self.module_name)), 'wb') as f:
        pickle.dump(state, f)

  def load(self, save_folder):
    load_path = os.path.join(save_folder, "{}.pickle".format(self.module_name))
    if os.path.exists(load_path):
      with open(load_path, 'rb') as f:
        state = pickle.load(f)
      self.buffer._set_state(state)
    else:
      self._agent.logger.log_color('###############################################################', '', color='red')
      self._agent.logger.log_color('WARNING', 'Replay buffer is not being loaded / was not saved.', color='cyan')
      self._agent.logger.log_color('WARNING', 'Replay buffer is not being loaded / was not saved.', color='red')
      self._agent.logger.log_color('WARNING', 'Replay buffer is not being loaded / was not saved.', color='yellow')
      self._agent.logger.log_color('###############################################################', '', color='red')
