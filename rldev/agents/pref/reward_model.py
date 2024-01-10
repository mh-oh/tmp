
import copy
import numpy as np
import pickle
import torch as th
import time

from collections import OrderedDict
from gymnasium import spaces
from itertools import combinations
from overrides import overrides
from pathlib import Path
from typing import *

from torch import nn
from torch.nn import functional as F

from rldev.agents.core import Node, Agent
from rldev.agents.pref.models import FusionMLP
from rldev.buffers.basic import EpisodicDictBuffer
from rldev.utils import torch as thu
from rldev.utils.env import *
from rldev.utils.structure import stack


device = 'cuda'

def KCenterGreedy(obs, full_obs, num_new_sample):
  selected_index = []
  current_index = list(range(obs.shape[0]))
  new_obs = obs
  new_full_obs = full_obs
  start_time = time.time()
  for count in range(num_new_sample):
    dist = compute_smallest_dist(new_obs, new_full_obs)
    max_index = th.argmax(dist)
    max_index = max_index.item()
    
    if count == 0:
      selected_index.append(max_index)
    else:
      selected_index.append(current_index[max_index])
    current_index = current_index[0:max_index] + current_index[max_index+1:]
    
    new_obs = obs[current_index]
    new_full_obs = np.concatenate([
        full_obs, 
        obs[selected_index]], 
        axis=0)
  return selected_index


def compute_smallest_dist(obs, full_obs):
  obs = th.from_numpy(obs).float()
  full_obs = th.from_numpy(full_obs).float()
  batch_size = 100
  with th.no_grad():
    total_dists = []
    for full_idx in range(len(obs) // batch_size + 1):
      full_start = full_idx * batch_size
      if full_start < len(obs):
        full_end = (full_idx + 1) * batch_size
        dists = []
        for idx in range(len(full_obs) // batch_size + 1):
          start = idx * batch_size
          if start < len(full_obs):
            end = (idx + 1) * batch_size
            dist = th.norm(
                obs[full_start:full_end, None, :].to(device) - full_obs[None, start:end, :].to(device), dim=-1, p=2
            )
            dists.append(dist)
        dists = th.cat(dists, dim=1)
        small_dists = th.th.min(dists, dim=1).values
        total_dists.append(small_dists)
            
    total_dists = th.cat(total_dists)
  return total_dists.unsqueeze(1)


class RewardModel(Node):

  def __init__(self, 
               agent: Agent,
               observation_space: spaces.Dict, 
               action_space: spaces.Box, 
               max_episode_steps: int,
               fusion: int,
               lr: float = 3e-4, 
               batch_size: int = 128,
               budget: int = 128, 
               segment_length: int = 1, 
               max_episodes: int = 100, 
               capacity: int = int(5e5),  
               label_margin=0.0, 
               teacher_beta=-1, 
               teacher_gamma=1, 
               teacher_eps_mistake=0, 
               teacher_eps_skip=0, 
               teacher_eps_equal=0):
    super().__init__(agent)

    odim = observation_dim(observation_space)
    adim = action_dim(action_space)

    self._observation_space = observation_space
    self._action_space = action_space

    self._fusion = fusion
    self._segment_length = segment_length = min(max_episode_steps, segment_length)
    self._max_episode_steps = max_episode_steps
    self._batch_size = batch_size
    self._budget = budget
    self._effective_budget = budget

    self._capacity = capacity
    self._cursor = 0
    self._full = False

    self._feedbacks_1 = np.empty((capacity,), dtype=object)
    self._feedbacks_2 = np.empty((capacity,), dtype=object)
    self._feedbacks_y = np.empty((capacity, 1), dtype=np.float32)
    
    self._r = FusionMLP(observation_space,
                        action_space,
                        fusion=fusion, lr=lr)

    self._teacher_beta = teacher_beta
    self._teacher_gamma = teacher_gamma
    self._teacher_eps_mistake = teacher_eps_mistake
    self._teacher_eps_equal = teacher_eps_equal
    self._teacher_eps_skip = teacher_eps_skip
    self._teacher_thres_skip = 0
    self._teacher_thres_equal = 0
    
    self._label_margin = label_margin
    self._label_target = 1 - 2 * label_margin

    self._buffer = (
      EpisodicDictBuffer(agent,
                         agent._env.num_envs,
                         (max_episodes + 1) * max_episode_steps,
                         observation_space,
                         action_space))

  def softXEnt_loss(self, input, target):
    logprobs = th.nn.functional.log_softmax (input, dim = 1)
    return  -(target * logprobs).sum() / input.shape[0]
  
  def change_batch(self, new_frac):
    self._effective_budget = int(self._budget*new_frac)
  
  def set_batch(self, new_batch):
    self._effective_budget = int(new_batch)
      
  def set_teacher_thres_skip(self, new_margin):
    self._teacher_thres_skip = new_margin * self._teacher_eps_skip
      
  def set_teacher_thres_equal(self, new_margin):
    self._teacher_thres_equal = new_margin * self._teacher_eps_equal
  
  def add(self,
          observation: Dict,
          action: np.ndarray,
          reward: np.ndarray,
          next_observation: Dict,
          done: np.ndarray):

    self._buffer.add(observation,
                     action,
                     reward,
                     next_observation,
                     done,
                     done,
                     {})

  def _r_member(self, observation, action, member):
    observation = flatten_observation(self._observation_space,
                                      observation)
    return self._r[member](thu.torch(np.concatenate([observation, action], axis=-1)))

  # def get_rank_probability(self, x_1, x_2):
  #   # get probability x_1 > x_2
  #   probs = []
  #   for member in range(self._fusion):
  #     probs.append(self.p_hat_member(x_1, x_2, member=member).cpu().numpy())
  #   probs = np.array(probs)
    
  #   return np.mean(probs, axis=0), np.std(probs, axis=0)
  
  def get_entropy(self, first, second):
    # get probability x_1 > x_2
    probs = []
    for member in range(self._fusion):
      probs.append(self.p_hat_entropy(first, second, member=member).cpu().numpy())
    probs = np.array(probs)
    return np.mean(probs, axis=0), np.std(probs, axis=0)

  # def p_hat_member(self, x_1, x_2, member=-1):
  #   # softmaxing to get the probabilities according to eqn 1
  #   with th.no_grad():
  #     r_hat1 = self._r_member(x_1, member) #self._r[member](thu.torch(x_1))
  #     r_hat2 = self._r_member(x_2, member) #self._r[member](thu.torch(x_2))
  #     r_hat1 = r_hat1.sum(axis=1)
  #     r_hat2 = r_hat2.sum(axis=1)
  #     r_hat = th.cat([r_hat1, r_hat2], axis=-1)
    
  #   # taking 0 index for probability x_1 > x_2
  #   return F.softmax(r_hat, dim=-1)[:,0]
  
  def p_hat_entropy(self, first, second, member=-1):
    # softmaxing to get the probabilities according to eqn 1
    with th.no_grad():
      r_hat1 = self._r_member(*self._stack(first), member) #self._r[member](thu.torch(x_1))
      r_hat2 = self._r_member(*self._stack(second), member) #self._r[member](thu.torch(x_2))
      r_hat1 = r_hat1.sum(axis=1)
      r_hat2 = r_hat2.sum(axis=1)
      r_hat = th.cat([r_hat1, r_hat2], axis=-1)
    
    ent = F.softmax(r_hat, dim=-1) * F.log_softmax(r_hat, dim=-1)
    ent = ent.sum(axis=-1).abs()
    return ent

  def r_hat(self,
            observation: OrderedDict,
            action: np.ndarray):
    observation = flatten_observation(
      self._observation_space, observation)
    return self._r(th.from_numpy(np.concatenate([observation, action], axis=-1)).float().to(device), reduce="mean")
  
  @overrides
  def save(self, dir: Path):
    dir.mkdir(parents=True, exist_ok=True)
    
    th.save(self._r.optimizer, dir / "_r_optimizer.pt")
    th.save(self._r, dir / "_r.pt")

    def save_nosync(file, obj):
      with open(dir / f"{file}.npy.nosync", "wb") as fout:
        np.save(fout, obj)
    
    save_nosync("_feedbacks_1", self._feedbacks_1)
    save_nosync("_feedbacks_2", self._feedbacks_2)
    save_nosync("_feedbacks_y", self._feedbacks_y)

    with open(dir / "_cursor.pkl", "wb") as fout:
      pickle.dump(self._cursor, fout)
    with open(dir / "_full.pkl", "wb") as fout:
      pickle.dump(self._full, fout)

  @overrides
  def load(self, dir: Path):
    ...
  
  def get_train_acc(self):
    ensemble_acc = np.array([0 for _ in range(self._fusion)])
    max_len = self._capacity if self._full else self._cursor
    batch_size = 256
    num_epochs = int(np.ceil(max_len/batch_size))
    
    total = 0
    for epoch in range(num_epochs):
      last_index = (epoch+1)*batch_size
      if (epoch+1)*batch_size > max_len:
        last_index = max_len

      first, second, y = (self._feedbacks_1[epoch*batch_size:last_index],
                          self._feedbacks_2[epoch*batch_size:last_index],
                          self._feedbacks_y[epoch*batch_size:last_index])

      observations_1, actions_1 = self._stack(first)
      observations_2, actions_2 = self._stack(second)
      labels = th.from_numpy(y.flatten()).long().to(device)

      total += labels.size(0)
      for member in range(self._fusion):
        # get logits
        r_hat1 = self._r_member(observations_1, actions_1, member) #self._r[member](thu.torch(sa_t_1))
        r_hat2 = self._r_member(observations_2, actions_2, member) #self._r[member](thu.torch(sa_t_2))
        r_hat1 = r_hat1.sum(axis=1)
        r_hat2 = r_hat2.sum(axis=1)
        r_hat = th.cat([r_hat1, r_hat2], axis=-1)                
        _, predicted = th.max(r_hat.data, 1)
        correct = (predicted == labels).sum().item()
        ensemble_acc[member] += correct
            
    ensemble_acc = ensemble_acc / total
    return np.mean(ensemble_acc)

  def query(self, mode, **kwargs):

    query = getattr(self, f"_query_{mode}")
    first, second, label = self.answer(
      *query(self._effective_budget, **kwargs))
    if len(label) > 0:
      self.store_feedbacks(first, second, label)
    return len(label)

  u"""Acquisition functions.
  They return batched pair of segments."""

  def _query_uniform(self, n):
    u"""Random uniform."""
    return self._random_pairs(
      self._episodes(), n, self._segment_length)

  def _query_uniform_aligned(self, 
                             n,
                             *,
                             cluster,
                             cluster_discard_outlier=True):
    u"""Random uniform pairs of segments with matching targets."""

    from rldev.utils.structure import chunk
    labels, clusters = self._compute_clusters(
      self._episodes(), cluster, cluster_discard_outlier)
    
    pairs = []
    for cluster, k in zip(clusters,
                          chunk(n, len(labels))):
      pairs.append([
        *self._random_pairs(cluster, 
                            k, 
                            self._segment_length)])
    pairs = np.array(pairs, dtype=object)
    return (np.concatenate(pairs[:, 0], axis=0),
            np.concatenate(pairs[:, 1], axis=0))
  
  def _query_entropy(self, n, *, scale):
    u"""Entropy."""

    first, second = self._random_pairs(self._episodes(), n * scale)

    entropy, _ = self.get_entropy(first, second)
    
    topn = (-entropy).argsort()[:n]
    return first[topn], second[topn]

  def _query_entropy_aligned(self, 
                             n,
                             *,
                             cluster,
                             cluster_discard_outlier=True):
    u"""Top-`n` entropy pairs with matching targets."""

    first, second = self._every_aligned_pairs(
      self._episodes(), self._segment_length, cluster, cluster_discard_outlier)
    

    entropy, _ = self.get_entropy(first, second)
    
    topn = (-entropy).argsort()[:n]
    return first[topn], second[topn]

  u"""Common auxiliary functions."""

  def _episodes(self):
    return np.array(
      list(self._buffer.get_episodes()), dtype=object)

  def _sample(self, episodes, n):
    u"""Sample `n` samples randomly from `episodes`."""
    return episodes[
      np.random.choice(len(episodes), size=n, replace=True)]

  def _random_pairs(self, episodes, n, segment_length):
    u"""Sample `n` random pairs of segments."""
    first, second = (self._segment(episodes, size=segment_length),
                     self._segment(episodes, size=segment_length))
    return (self._sample(first, n), 
            self._sample(second, n))

  def _discover_targets(self, episodes):
    u"""Extranct targets from `episodes`."""
    targets = []
    for episode in episodes:
      target, = np.unique(
        episode.observation["desired_goal"], axis=0)
      targets.append(target)
    return np.array(targets)

  def _compute_clusters(self, 
                        episodes, 
                        cluster,
                        cluster_discard_outlier):
    u"""Cluster `episodes` based on their desired goals."""

    labels = cluster.fit(self._discover_targets(episodes)).labels_
    unique_labels = set(labels)
    if cluster_discard_outlier:
      unique_labels.discard(-1)

    unique_labels = list(unique_labels)
    return (unique_labels, list(episodes[labels == label] 
                                for label in unique_labels))

  def _every_pairs(self, episodes):
    u"""Find every pairs of trajectories."""
    pairs = np.array([[first, second]
      for first, second in combinations(episodes, 2)], dtype=object)
    return pairs[..., 0], pairs[..., 1]

  def _every_aligned_pairs(self,
                           episodes,
                           length,
                           cluster,
                           cluster_discard_outlier):
    u"""Find every pairs of trajectories with matching 
    desired goals."""

    _, clusters = self._compute_clusters(
      episodes, cluster, cluster_discard_outlier)

    pairs = []
    for cluster in clusters:
      pairs.append([
        *self._every_pairs(self._segment(cluster, length))])    
    pairs = np.array(pairs, dtype=object)
    return (np.concatenate(pairs[:, 0], axis=0),
            np.concatenate(pairs[:, 1], axis=0))

  def _segment(self, episodes, size):
    u"""Randomly segment `episodes` with size `size`."""
    def get(episode):
      steps = (np.arange(0, size) + 
               np.random.randint(
                 0, self._max_episode_steps - size + 1))
      return episode.get(steps)
    return np.array([get(episode) 
                     for episode in episodes], dtype=object)

  def _compat(self, first, second):

    sa_t_1, r_t_1 = self._unpack(first)
    sa_t_2, r_t_2 = self._unpack(second)

    return sa_t_1, sa_t_2, r_t_1, r_t_2

  def _unpack(self, segments):

    def fn(observation):
      env = self.agent._env
      return flatten_observation(env.envs[0].observation_space,
                                 observation)

    sa_t, r_t = [], []
    for seg in segments:
      sa_t.append(np.concatenate([fn(seg.observation), seg.action], axis=-1))
      r_t.append(seg.reward)
    return np.stack(sa_t, axis=0), np.stack(r_t, axis=0)[..., np.newaxis]

  def _stack(self, segments):
    
    def get(fn):
      return [fn(s) for s in segments]
    return (stack(get(lambda s: s.observation), axis=0),
            stack(get(lambda s: s.action), axis=0))

  def store_feedbacks(self, first, second, labels):
    sa_t_1, sa_t_2, r_t_1, r_t_2 = self._compat(first, second)
    n = sa_t_1.shape[0]

    cursor = self._cursor
    index = np.arange(cursor, cursor + n) % self._capacity
    self._feedbacks_1[index] = copy.deepcopy(first)
    self._feedbacks_2[index] = copy.deepcopy(second)
    self._feedbacks_y[index] = copy.deepcopy(labels)
    self._cursor = (cursor + n) % self._capacity

    if cursor + n >= self._capacity:
      self._full = True

  def answer(self, first, second):
    sa_t_1, sa_t_2, r_t_1, r_t_2 = self._compat(first, second)

    assert len(sa_t_1) == self._effective_budget
    sum_r_t_1 = np.sum(r_t_1, axis=1)
    sum_r_t_2 = np.sum(r_t_2, axis=1)
    
    # skip the query
    if self._teacher_thres_skip > 0: 
      max_r_t = np.maximum(sum_r_t_1, sum_r_t_2)
      max_index = (max_r_t > self._teacher_thres_skip).reshape(-1)
      if sum(max_index) == 0:
        return None, None, None, None, []

      sa_t_1 = sa_t_1[max_index]
      sa_t_2 = sa_t_2[max_index]
      r_t_1 = r_t_1[max_index]
      r_t_2 = r_t_2[max_index]
      first = first[max_index]
      second = second[max_index]
      sum_r_t_1 = np.sum(r_t_1, axis=1)
      sum_r_t_2 = np.sum(r_t_2, axis=1)
    
    # equally preferable
    margin_index = (np.abs(sum_r_t_1 - sum_r_t_2) < self._teacher_thres_equal).reshape(-1)
    
    # perfectly rational
    seg_size = r_t_1.shape[1]
    temp_r_t_1 = r_t_1.copy()
    temp_r_t_2 = r_t_2.copy()
    for index in range(seg_size-1):
      temp_r_t_1[:,:index+1] *= self._teacher_gamma
      temp_r_t_2[:,:index+1] *= self._teacher_gamma
    sum_r_t_1 = np.sum(temp_r_t_1, axis=1)
    sum_r_t_2 = np.sum(temp_r_t_2, axis=1)
        
    rational_labels = 1*(sum_r_t_1 < sum_r_t_2)
    if self._teacher_beta > 0: # Bradley-Terry rational model
      r_hat = th.cat([th.Tensor(sum_r_t_1), 
                          th.Tensor(sum_r_t_2)], axis=-1)
      r_hat = r_hat*self._teacher_beta
      ent = F.softmax(r_hat, dim=-1)[:, 1]
      labels = th.bernoulli(ent).int().numpy().reshape(-1, 1)
    else:
      labels = rational_labels
    
    # making a mistake
    len_labels = labels.shape[0]
    rand_num = np.random.rand(len_labels)
    noise_index = rand_num <= self._teacher_eps_mistake
    labels[noise_index] = 1 - labels[noise_index]

    # equally preferable
    labels[margin_index] = -1 
    
    return first, second, labels

  def train_reward(self):
    ensemble_losses = [[] for _ in range(self._fusion)]
    ensemble_acc = np.array([0 for _ in range(self._fusion)])
    
    max_len = self._capacity if self._full else self._cursor
    total_batch_index = []
    for _ in range(self._fusion):
      total_batch_index.append(np.random.permutation(max_len))
    
    num_epochs = int(np.ceil(max_len/self._batch_size))
    total = 0
    
    for epoch in range(num_epochs):
      self._r.optimizer.zero_grad()
      loss = 0.0
      
      last_index = (epoch+1)*self._batch_size
      if last_index > max_len:
          last_index = max_len
          
      for member in range(self._fusion):
          
        # get random batch
        idxs = total_batch_index[member][epoch*self._batch_size:last_index]
        first, second, y = (self._feedbacks_1[idxs],
                            self._feedbacks_2[idxs],
                            self._feedbacks_y[idxs])

        observations_1, actions_1 = self._stack(first)
        observations_2, actions_2 = self._stack(second)
        
        labels = th.from_numpy(y.flatten()).long().to(device)
        
        if member == 0:
          total += labels.size(0)
        
        # get logits
        r_hat1 = self._r_member(observations_1, actions_1, member) #self._r[member](thu.torch(sa_t_1))
        r_hat2 = self._r_member(observations_2, actions_2, member) #self._r[member](thu.torch(sa_t_2))
        r_hat1 = r_hat1.sum(axis=1)
        r_hat2 = r_hat2.sum(axis=1)
        r_hat = th.cat([r_hat1, r_hat2], axis=-1)

        # compute loss
        curr_loss = nn.CrossEntropyLoss()(r_hat, labels)
        loss += curr_loss
        ensemble_losses[member].append(curr_loss.item())
        
        # compute acc
        _, predicted = th.max(r_hat.data, 1)
        correct = (predicted == labels).sum().item()
        ensemble_acc[member] += correct
          
      loss.backward()
      self._r.optimizer.step()
    
    ensemble_acc = ensemble_acc / total
    
    return ensemble_acc
  
  def train_soft_reward(self):
    ensemble_losses = [[] for _ in range(self._fusion)]
    ensemble_acc = np.array([0 for _ in range(self._fusion)])
    
    max_len = self._capacity if self._full else self._cursor
    total_batch_index = []
    for _ in range(self._fusion):
      total_batch_index.append(np.random.permutation(max_len))
    
    num_epochs = int(np.ceil(max_len/self._batch_size))
    list_debug_loss1, list_debug_loss2 = [], []
    total = 0
    
    for epoch in range(num_epochs):
      self._r.optimizer.zero_grad()
      loss = 0.0
      
      last_index = (epoch+1)*self._batch_size
      if last_index > max_len:
        last_index = max_len
          
      for member in range(self._fusion):
          
        # get random batch
        idxs = total_batch_index[member][epoch*self._batch_size:last_index]
        first, second, y = (self._feedbacks_1[idxs],
                            self._feedbacks_2[idxs],
                            self._feedbacks_y[idxs])

        observations_1, actions_1 = self._stack(first)
        observations_2, actions_2 = self._stack(second)

        labels = th.from_numpy(y.flatten()).long().to(device)
        
        if member == 0:
          total += labels.size(0)
        
        # get logits
        r_hat1 = self._r_member(observations_1, actions_1, member) #self._r[member](thu.torch(sa_t_1))
        r_hat2 = self._r_member(observations_2, actions_2, member) #self._r[member](thu.torch(sa_t_2))
        r_hat1 = r_hat1.sum(axis=1)
        r_hat2 = r_hat2.sum(axis=1)
        r_hat = th.cat([r_hat1, r_hat2], axis=-1)

        # compute loss
        uniform_index = labels == -1
        labels[uniform_index] = 0
        target_onehot = th.zeros_like(r_hat).scatter(1, labels.unsqueeze(1), self._label_target)
        target_onehot += self._label_margin
        if sum(uniform_index) > 0:
          target_onehot[uniform_index] = 0.5
        curr_loss = self.softXEnt_loss(r_hat, target_onehot)
        loss += curr_loss
        ensemble_losses[member].append(curr_loss.item())
        
        # compute acc
        _, predicted = th.max(r_hat.data, 1)
        correct = (predicted == labels).sum().item()
        ensemble_acc[member] += correct
          
      loss.backward()
      self._r.optimizer.step()
    
    ensemble_acc = ensemble_acc / total
    
    return ensemble_acc