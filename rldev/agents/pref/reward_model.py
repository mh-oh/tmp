
import numpy as np
import pickle
import torch
import torch as th
import time

from gymnasium import spaces
from itertools import combinations
from overrides import overrides
from pathlib import Path
from typing import *

from torch import nn
from torch.nn import functional as F

from rldev.agents.core import Node, Agent
from rldev.agents.pref.models import Fusion
from rldev.buffers.basic import EpisodicDictBuffer
from rldev.utils import gym_types
from rldev.utils import torch as thu
from rldev.utils.env import flatten_space, flatten_observation
from rldev.utils.nn import _MLP as MLP

device = 'cuda'

def KCenterGreedy(obs, full_obs, num_new_sample):
  selected_index = []
  current_index = list(range(obs.shape[0]))
  new_obs = obs
  new_full_obs = full_obs
  start_time = time.time()
  for count in range(num_new_sample):
    dist = compute_smallest_dist(new_obs, new_full_obs)
    max_index = torch.argmax(dist)
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
  obs = torch.from_numpy(obs).float()
  full_obs = torch.from_numpy(full_obs).float()
  batch_size = 100
  with torch.no_grad():
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
            dist = torch.norm(
                obs[full_start:full_end, None, :].to(device) - full_obs[None, start:end, :].to(device), dim=-1, p=2
            )
            dists.append(dist)
        dists = torch.cat(dists, dim=1)
        small_dists = torch.torch.min(dists, dim=1).values
        total_dists.append(small_dists)
            
    total_dists = torch.cat(total_dists)
  return total_dists.unsqueeze(1)


class RewardModel(Node):

  def __init__(self, 
               agent: Agent,
               observation_space: spaces.Dict, 
               action_space: spaces.Box, 
               max_episode_steps: int,
               fusion: int,
               activation: str,
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

    adim, = action_space.shape
    if isinstance(observation_space, gym_types.Box):
      odim, = observation_space.shape
    elif isinstance(observation_space, gym_types.Dict):
      odim, = flatten_space(observation_space).shape
    else:
      raise NotImplementedError()

    self._fusion = fusion
    self._segment_length = segment_length = min(max_episode_steps, segment_length)
    self._max_episode_steps = max_episode_steps
    self._batch_size = batch_size
    self._budget = budget
    self._effective_budget = budget
    
    self._capacity = capacity
    self._feedbacks_first = np.empty((capacity, segment_length, odim + adim), dtype=np.float32)
    self._feedbacks_second = np.empty((capacity, segment_length, odim + adim), dtype=np.float32)
    self._feedbacks_label = np.empty((capacity, 1), dtype=np.float32)
    self._cursor = 0
    self._full = False
    
    def thunk():
      return MLP(dims=[odim + adim, 256, 256, 256, 1],
                 activations=["leaky-relu",
                              "leaky-relu",
                              "leaky-relu",
                              activation]).float().to(device)

    self._r = Fusion([thunk for _ in range(fusion)])
    self._r_optimizer = torch.optim.Adam(self._r.parameters(), lr=lr)

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
    logprobs = torch.nn.functional.log_softmax (input, dim = 1)
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

  def get_rank_probability(self, x_1, x_2):
    # get probability x_1 > x_2
    probs = []
    for member in range(self._fusion):
      probs.append(self.p_hat_member(x_1, x_2, member=member).cpu().numpy())
    probs = np.array(probs)
    
    return np.mean(probs, axis=0), np.std(probs, axis=0)
  
  def get_entropy(self, x_1, x_2):
    # get probability x_1 > x_2
    probs = []
    for member in range(self._fusion):
      probs.append(self.p_hat_entropy(x_1, x_2, member=member).cpu().numpy())
    probs = np.array(probs)
    return np.mean(probs, axis=0), np.std(probs, axis=0)

  def p_hat_member(self, x_1, x_2, member=-1):
    # softmaxing to get the probabilities according to eqn 1
    with torch.no_grad():
      r_hat1 = self._r[member](thu.torch(x_1))
      r_hat2 = self._r[member](thu.torch(x_2))
      r_hat1 = r_hat1.sum(axis=1)
      r_hat2 = r_hat2.sum(axis=1)
      r_hat = torch.cat([r_hat1, r_hat2], axis=-1)
    
    # taking 0 index for probability x_1 > x_2
    return F.softmax(r_hat, dim=-1)[:,0]
  
  def p_hat_entropy(self, x_1, x_2, member=-1):
    # softmaxing to get the probabilities according to eqn 1
    with torch.no_grad():
      r_hat1 = self._r[member](thu.torch(x_1))
      r_hat2 = self._r[member](thu.torch(x_2))
      r_hat1 = r_hat1.sum(axis=1)
      r_hat2 = r_hat2.sum(axis=1)
      r_hat = torch.cat([r_hat1, r_hat2], axis=-1)
    
    ent = F.softmax(r_hat, dim=-1) * F.log_softmax(r_hat, dim=-1)
    ent = ent.sum(axis=-1).abs()
    return ent

  def r_hat(self, x):
    return self._r(th.from_numpy(x).float().to(device), reduce="mean")
  
  @overrides
  def save(self, dir: Path):
    dir.mkdir(parents=True, exist_ok=True)
    
    th.save(self._r_optimizer, dir / "_r_optimizer.pt")
    th.save(self._r, dir / "_r.pt")

    def save_nosync(file, obj):
      with open(dir / f"{file}.npy.nosync", "wb") as fout:
        np.save(fout, obj)
    
    save_nosync("_feedbacks_first", self._feedbacks_first)
    save_nosync("_feedbacks_second", self._feedbacks_second)
    save_nosync("_feedbacks_label", self._feedbacks_label)

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
          
      sa_t_1 = self._feedbacks_first[epoch*batch_size:last_index]
      sa_t_2 = self._feedbacks_second[epoch*batch_size:last_index]
      labels = self._feedbacks_label[epoch*batch_size:last_index]
      labels = torch.from_numpy(labels.flatten()).long().to(device)
      total += labels.size(0)
      for member in range(self._fusion):
        # get logits
        r_hat1 = self._r[member](thu.torch(sa_t_1))
        r_hat2 = self._r[member](thu.torch(sa_t_2))
        r_hat1 = r_hat1.sum(axis=1)
        r_hat2 = r_hat2.sum(axis=1)
        r_hat = torch.cat([r_hat1, r_hat2], axis=-1)                
        _, predicted = torch.max(r_hat.data, 1)
        correct = (predicted == labels).sum().item()
        ensemble_acc[member] += correct
            
    ensemble_acc = ensemble_acc / total
    return np.mean(ensemble_acc)

  def query(self, mode, **kwargs):

    fn = getattr(self, f"_query_{mode}")
    first, second = fn(self._effective_budget, **kwargs)
    sa_t_1, sa_t_2, r_t_1, r_t_2 = self._compat(first, second)
    # get labels
    sa_t_1, sa_t_2, r_t_1, r_t_2, labels = self.get_label(
        sa_t_1, sa_t_2, r_t_1, r_t_2)
    if len(labels) > 0:
      self.put_queries(sa_t_1, sa_t_2, labels)
    return len(labels)

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
    clusters = self._compute_clusters(
      self._episodes(), cluster, cluster_discard_outlier)
  
    pairs = []
    for (_, cluster), k in zip(clusters, 
                               chunk(n, len(clusters))):
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
    sa_t_1, sa_t_2, r_t_1, r_t_2 = self._compat(first, second)

    entropy, _ = self.get_entropy(sa_t_1, sa_t_2)
    
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
    
    sa_t_1, sa_t_2, r_t_1, r_t_2 = self._compat(first, second)

    entropy, _ = self.get_entropy(sa_t_1, sa_t_2)
    
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
    def segments():
      return self._segment(episodes, size=segment_length)
    return (self._sample(segments(), n), 
            self._sample(segments(), n))

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

    targets = self._discover_targets(episodes)
    labels = cluster.fit(targets).labels_
    if labels.ndim != 1:
      raise AssertionError(f"{labels.ndim} != 1")

    index = np.argsort(labels)
    unique, (_, *sections) = np.unique(labels[index],
                                       return_index=True)
    labels, clusters = (unique.tolist(),
                        np.split(episodes[index], sections))
    res = []
    for label, cluster in zip(labels, clusters):
      if not (cluster_discard_outlier and label == -1):
        res.append((label, cluster))
    return res

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

    clusters = self._compute_clusters(
      episodes, cluster, cluster_discard_outlier)

    pairs = np.array([
      [*self._every_pairs(self._segment(cluster, length))] 
        for _, cluster in clusters], dtype=object)
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

    def fn(observation):
      env = self.agent._env
      return flatten_observation(env.envs[0].observation_space,
                                 observation)

    def _compat(segments):
      sa_t, r_t = [], []
      for seg in segments:
        sa_t.append(np.concatenate([fn(seg.observation), seg.action], axis=-1))
        r_t.append(seg.reward)
      return np.stack(sa_t, axis=0), np.stack(r_t, axis=0)[..., np.newaxis]

    sa_t_1, r_t_1 = _compat(first)
    sa_t_2, r_t_2 = _compat(second)

    return sa_t_1, sa_t_2, r_t_1, r_t_2

  def put_queries(self, sa_t_1, sa_t_2, labels):
    total_sample = sa_t_1.shape[0]
    next_index = self._cursor + total_sample
    if next_index >= self._capacity:
      self._full = True
      maximum_index = self._capacity - self._cursor
      np.copyto(self._feedbacks_first[self._cursor:self._capacity], sa_t_1[:maximum_index])
      np.copyto(self._feedbacks_second[self._cursor:self._capacity], sa_t_2[:maximum_index])
      np.copyto(self._feedbacks_label[self._cursor:self._capacity], labels[:maximum_index])

      remain = total_sample - (maximum_index)
      if remain > 0:
        np.copyto(self._feedbacks_first[0:remain], sa_t_1[maximum_index:])
        np.copyto(self._feedbacks_second[0:remain], sa_t_2[maximum_index:])
        np.copyto(self._feedbacks_label[0:remain], labels[maximum_index:])

      self._cursor = remain
    else:
      np.copyto(self._feedbacks_first[self._cursor:next_index], sa_t_1)
      np.copyto(self._feedbacks_second[self._cursor:next_index], sa_t_2)
      np.copyto(self._feedbacks_label[self._cursor:next_index], labels)
      self._cursor = next_index
          
  def get_label(self, sa_t_1, sa_t_2, r_t_1, r_t_2):
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
      r_hat = torch.cat([torch.Tensor(sum_r_t_1), 
                          torch.Tensor(sum_r_t_2)], axis=-1)
      r_hat = r_hat*self._teacher_beta
      ent = F.softmax(r_hat, dim=-1)[:, 1]
      labels = torch.bernoulli(ent).int().numpy().reshape(-1, 1)
    else:
      labels = rational_labels
    
    # making a mistake
    len_labels = labels.shape[0]
    rand_num = np.random.rand(len_labels)
    noise_index = rand_num <= self._teacher_eps_mistake
    labels[noise_index] = 1 - labels[noise_index]

    # equally preferable
    labels[margin_index] = -1 
    
    return sa_t_1, sa_t_2, r_t_1, r_t_2, labels

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
      self._r_optimizer.zero_grad()
      loss = 0.0
      
      last_index = (epoch+1)*self._batch_size
      if last_index > max_len:
          last_index = max_len
          
      for member in range(self._fusion):
          
        # get random batch
        idxs = total_batch_index[member][epoch*self._batch_size:last_index]
        sa_t_1 = self._feedbacks_first[idxs]
        sa_t_2 = self._feedbacks_second[idxs]
        labels = self._feedbacks_label[idxs]
        labels = torch.from_numpy(labels.flatten()).long().to(device)
        
        if member == 0:
          total += labels.size(0)
        
        # get logits
        r_hat1 = self._r[member](thu.torch(sa_t_1))
        r_hat2 = self._r[member](thu.torch(sa_t_2))
        r_hat1 = r_hat1.sum(axis=1)
        r_hat2 = r_hat2.sum(axis=1)
        r_hat = torch.cat([r_hat1, r_hat2], axis=-1)

        # compute loss
        curr_loss = nn.CrossEntropyLoss()(r_hat, labels)
        loss += curr_loss
        ensemble_losses[member].append(curr_loss.item())
        
        # compute acc
        _, predicted = torch.max(r_hat.data, 1)
        correct = (predicted == labels).sum().item()
        ensemble_acc[member] += correct
          
      loss.backward()
      self._r_optimizer.step()
    
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
      self._r_optimizer.zero_grad()
      loss = 0.0
      
      last_index = (epoch+1)*self._batch_size
      if last_index > max_len:
        last_index = max_len
          
      for member in range(self._fusion):
          
        # get random batch
        idxs = total_batch_index[member][epoch*self._batch_size:last_index]
        sa_t_1 = self._feedbacks_first[idxs]
        sa_t_2 = self._feedbacks_second[idxs]
        labels = self._feedbacks_label[idxs]
        labels = torch.from_numpy(labels.flatten()).long().to(device)
        
        if member == 0:
          total += labels.size(0)
        
        # get logits
        r_hat1 = self._r[member](thu.torch(sa_t_1))
        r_hat2 = self._r[member](thu.torch(sa_t_2))
        r_hat1 = r_hat1.sum(axis=1)
        r_hat2 = r_hat2.sum(axis=1)
        r_hat = torch.cat([r_hat1, r_hat2], axis=-1)

        # compute loss
        uniform_index = labels == -1
        labels[uniform_index] = 0
        target_onehot = torch.zeros_like(r_hat).scatter(1, labels.unsqueeze(1), self._label_target)
        target_onehot += self._label_margin
        if sum(uniform_index) > 0:
          target_onehot[uniform_index] = 0.5
        curr_loss = self.softXEnt_loss(r_hat, target_onehot)
        loss += curr_loss
        ensemble_losses[member].append(curr_loss.item())
        
        # compute acc
        _, predicted = torch.max(r_hat.data, 1)
        correct = (predicted == labels).sum().item()
        ensemble_acc[member] += correct
          
      loss.backward()
      self._r_optimizer.step()
    
    ensemble_acc = ensemble_acc / total
    
    return ensemble_acc