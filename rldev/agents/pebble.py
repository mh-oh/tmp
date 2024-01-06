
import numpy as np

from overrides import overrides
from rldev.agents.pref.agent import PbRLAgent


class PEBBLE(PbRLAgent):

  def __init__(self,
               config,
               env,
               test_env,
               policy,
               buffer,
               reward_model):
    super().__init__(config,
                     env,
                     test_env,
                     policy,
                     buffer,
                     reward_model)

  @overrides
  def save(self): super().save()
  
  @overrides
  def load(self): super().load()

  @overrides
  def process_episodic_records(self, done):
    return super().process_episodic_records(done)

  @overrides
  def optimize_reward_model(self):

    # To update reward schedule
    def frac(step):
      training_steps = self._training_steps
      reward_schedule = self.config.reward_schedule
      if reward_schedule == 1:
        frac = (training_steps - step) / training_steps
        if frac == 0:
          frac = 0.01
      elif reward_schedule == 2:
        frac = training_steps / (training_steps - step + 1)
      else:
        frac = 1
      return frac

    # run training update
    if self._step == self.config.num_seed_steps + self.config.num_unsup_steps:
      self._reward_model.change_batch(frac(self._step))
      
      # update margin --> not necessary / will be updated soon
      new_margin = np.mean(self._episode_returns) * (self.config.segment_length / self._env._max_episode_steps)
      self._reward_model.set_teacher_thres_skip(new_margin)
      self._reward_model.set_teacher_thres_equal(new_margin)
      
      # first learn reward
      self.learn_reward(first_flag=1)
      
      # relabel buffer
      self._buffer.relabel_rewards(self._reward_model)
      
      # reset interact_count
      self._interact_count = 0
    if self._step > self.config.num_seed_steps + self.config.num_unsup_steps:
      # update reward function
      if ((self._feedbacks < self.config.max_feedback) and
          (self._interact_count % self.config.num_interact == 0)):
        self._reward_model.change_batch(frac(self._step))
        
        # update margin --> not necessary / will be updated soon
        new_margin = np.mean(self._episode_returns) * (self.config.segment_length / self._env._max_episode_steps)
        self._reward_model.set_teacher_thres_skip(new_margin * self.config.teacher_eps_skip)
        self._reward_model.set_teacher_thres_equal(new_margin * self.config.teacher_eps_equal)
        
        # corner case: new total feed > max feed
        if self._reward_model.mb_size + self._feedbacks > self.config.max_feedback:
          self._reward_model.set_batch(self.config.max_feedback - self._feedbacks)
            
        self.learn_reward()
        self._buffer.relabel_rewards(self._reward_model)

  @overrides
  def optimize_policy(self):

    # unsupervised exploration
    if ((self._step > self.config.num_seed_steps) and 
        (self._step < self.config.num_seed_steps + self.config.num_unsup_steps)):
      self._policy.update_state_ent(self._buffer, self.logger, self._step, 
                                  gradient_update=1, K=self.config.topK)

    # run training update
    if self._step == self.config.num_seed_steps + self.config.num_unsup_steps:
      # reset Q due to unsuperivsed exploration
      self._policy.reset_critic()
      # update agent
      self._policy.update(
          self._buffer, self.logger, self._step, 
          gradient_update=self.config.reset_update)

    if self._step > self.config.num_seed_steps + self.config.num_unsup_steps:
      self._policy.update(self._buffer, self.logger, self._step, 1)

  def learn_reward(self, first_flag=0):

    conf = self.config
    def query():
      fn = self._reward_model.query
      if first_flag == 1:
        return fn("uniform")
      else:
        return fn(conf.query.mode, **conf.query.kwargs)
    
    self._feedbacks += self._reward_model._budget
    self._labeled_feedbacks += query()
    
    train_acc = 0
    if self._labeled_feedbacks > 0:
      # update reward
      for epoch in range(self.config.reward_update):
        if self.config.label_margin > 0 or self.config.teacher_eps_equal > 0:
          train_acc = self._reward_model.train_soft_reward()
        else:
          train_acc = self._reward_model.train_reward()
        total_acc = np.mean(train_acc)
        
        if total_acc > 0.97:
          break
                
    print("Reward function is updated!! ACC: " + str(total_acc))