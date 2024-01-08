
import numpy as np
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import time

from gymnasium import spaces
from itertools import combinations, repeat
from overrides import overrides
from pathlib import Path
from typing import *

from rldev.agents.core import Node, Agent
from rldev.agents.pref.models import Fusion
from rldev.buffers.basic import EpisodicDictBuffer
from rldev.utils import gym_types
from rldev.utils import torch as thu
from rldev.utils.env import flatten_space, flatten_observation
from rldev.utils.structure import isiterable, pairwise

device = 'cuda'


class MLP(nn.Module):

  def __init__(self, dims, activations):
    super().__init__()
    
    if not isiterable(dims):
      raise ValueError(f"'dims' should be iterable")

    layers = len(dims) - 1
    if not isiterable(activations):
      activations = repeat(activations, times=layers)

    def map(x):
      if not isinstance(x, str):
        return x
      return {"leaky-relu": nn.LeakyReLU,
              "tanh": nn.Tanh,
              "sigmoid": nn.Sigmoid,
              "relu": nn.ReLU}[x]
    activations = [map(x) for x in activations]

    structure = []
    for (isize, osize), cls in zip(pairwise(dims), activations):
      structure.extend((nn.Linear(isize, osize),
                        cls()))
    self.body = nn.Sequential(*structure)

  def forward(self, input):
    return self.body(input)


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
               budget: int = 128, 
               segment_length: int = 1, 
               max_episodes: int = 100, 
               capacity: float = 5e5,  
               label_margin=0.0, 
               teacher_beta=-1, 
               teacher_gamma=1, 
               teacher_eps_mistake=0, 
               teacher_eps_skip=0, 
               teacher_eps_equal=0):
    super().__init__(agent)

    da, = action_space.shape
    if isinstance(observation_space, gym_types.Box):
      ds, = observation_space.shape
    elif isinstance(observation_space, gym_types.Dict):
      ds, = flatten_space(observation_space).shape

    # train data is trajectories, must process to sa and s..   
    self.ds = ds
    self.da = da
    self.de = fusion
    self.lr = lr
    self.max_episodes = max_episodes
    self.activation = activation
    if segment_length > max_episode_steps:
      segment_length = max_episode_steps
    self.size_segment = segment_length
    self._max_episode_steps = max_episode_steps
    
    self.capacity = int(capacity)
    self.buffer_seg1 = np.empty((self.capacity, segment_length, self.ds+self.da), dtype=np.float32)
    self.buffer_seg2 = np.empty((self.capacity, segment_length, self.ds+self.da), dtype=np.float32)
    self.buffer_label = np.empty((self.capacity, 1), dtype=np.float32)
    self.buffer_index = 0
    self.buffer_full = False
    
    def thunk():
      return MLP(dims=[self.ds + self.da, 256, 256, 256, 1],
                 activations=["leaky-relu",
                              "leaky-relu",
                              "leaky-relu",
                              self.activation]).float().to(device)

    self._r = Fusion([thunk for _ in range(self.de)])
    self._r_optimizer = torch.optim.Adam(self._r.parameters(), lr=self.lr)

    self._effective_budget = budget
    self._budget = budget
    self.train_batch_size = 128
    self.CEloss = nn.CrossEntropyLoss()
    
    # new teacher
    self.teacher_beta = teacher_beta
    self.teacher_gamma = teacher_gamma
    self.teacher_eps_mistake = teacher_eps_mistake
    self.teacher_eps_equal = teacher_eps_equal
    self.teacher_eps_skip = teacher_eps_skip
    self.teacher_thres_skip = 0
    self.teacher_thres_equal = 0
    
    self.label_margin = label_margin
    self.label_target = 1 - 2*self.label_margin

    ####

    self._buffer = EpisodicDictBuffer(agent,
                                      agent._env.num_envs,
                                      (max_episodes + 1) * max_episode_steps,
                                      observation_space,
                                      action_space)

  def softXEnt_loss(self, input, target):
    logprobs = torch.nn.functional.log_softmax (input, dim = 1)
    return  -(target * logprobs).sum() / input.shape[0]
  
  def change_batch(self, new_frac):
    self._effective_budget = int(self._budget*new_frac)
  
  def set_batch(self, new_batch):
    self._effective_budget = int(new_batch)
      
  def set_teacher_thres_skip(self, new_margin):
    self.teacher_thres_skip = new_margin * self.teacher_eps_skip
      
  def set_teacher_thres_equal(self, new_margin):
    self.teacher_thres_equal = new_margin * self.teacher_eps_equal
  
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
    for member in range(self.de):
      probs.append(self.p_hat_member(x_1, x_2, member=member).cpu().numpy())
    probs = np.array(probs)
    
    return np.mean(probs, axis=0), np.std(probs, axis=0)
  
  def get_entropy(self, x_1, x_2):
    # get probability x_1 > x_2
    probs = []
    for member in range(self.de):
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

  @overrides
  def load(self, dir: Path):
    ...
  
  def get_train_acc(self):
    ensemble_acc = np.array([0 for _ in range(self.de)])
    max_len = self.capacity if self.buffer_full else self.buffer_index
    total_batch_index = np.random.permutation(max_len)
    batch_size = 256
    num_epochs = int(np.ceil(max_len/batch_size))
    
    total = 0
    for epoch in range(num_epochs):
      last_index = (epoch+1)*batch_size
      if (epoch+1)*batch_size > max_len:
        last_index = max_len
          
      sa_t_1 = self.buffer_seg1[epoch*batch_size:last_index]
      sa_t_2 = self.buffer_seg2[epoch*batch_size:last_index]
      labels = self.buffer_label[epoch*batch_size:last_index]
      labels = torch.from_numpy(labels.flatten()).long().to(device)
      total += labels.size(0)
      for member in range(self.de):
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

  def _episodes(self):
    return np.array(
      list(self._buffer.get_episodes()), dtype=object)

  def query(self, mode, **kwargs):

    fn = getattr(self, f"_query_{mode}")
    return fn(self._effective_budget, **kwargs)

  def _random_pairs(self, episodes, n, segment_length):

    first = self._segment(episodes, size=segment_length)
    second = self._segment(episodes, size=segment_length)
    return self._sample(first, n), self._sample(second, n)

  def _query_uniform_aligned(self, 
                             n,
                             *,
                             cluster,
                             cluster_discard_outlier=True):

    episodes = self._episodes()

    goals = []
    for episode in episodes:
      y = np.unique(episode.observation["desired_goal"], axis=0)
      assert len(y) == 1
      goals.append(y[0])
    goals = np.array(goals)

    cluster = cluster.fit(goals)

    from rldev.utils.structure import chunk

    _sa_t_1, _sa_t_2, _r_t_1, _r_t_2 = [], [], [], []
    labels = set(cluster.labels_)
    if cluster_discard_outlier:
      print("discard")
      labels.discard(-1)
    print(labels)
    for x in chunk(range(n), len(labels)):
      g = labels.pop()

      mask = cluster.labels_ == g
      sa_t_1, sa_t_2, r_t_1, r_t_2 = self._compat(*self._random_pairs(episodes[mask], len(x), self.size_segment))
      print(n, len(sa_t_1))
      _sa_t_1.append(sa_t_1)
      _sa_t_2.append(sa_t_2)
      _r_t_1.append(r_t_1)
      _r_t_2.append(r_t_2)
    
    sa_t_1 = np.concatenate(_sa_t_1, axis=0)
    sa_t_2 = np.concatenate(_sa_t_2, axis=0)
    r_t_1 = np.concatenate(_r_t_1, axis=0)
    r_t_2 = np.concatenate(_r_t_2, axis=0)

    # get labels
    sa_t_1, sa_t_2, r_t_1, r_t_2, labels = self.get_label(
        sa_t_1, sa_t_2, r_t_1, r_t_2)
    
    if len(labels) > 0:
      self.put_queries(sa_t_1, sa_t_2, labels)
    
    return len(labels)

  def _query_uniform(self, n):

    episodes = self._episodes()

    sa_t_1, sa_t_2, r_t_1, r_t_2 = self._compat(*self._random_pairs(episodes, n, self.size_segment))

    # get labels
    sa_t_1, sa_t_2, r_t_1, r_t_2, labels = self.get_label(
        sa_t_1, sa_t_2, r_t_1, r_t_2)
    
    if len(labels) > 0:
      self.put_queries(sa_t_1, sa_t_2, labels)
    
    return len(labels)

  def _sample(self, episodes, n):
    u"""Sample `n` samples randomly from `episodes`."""
    return episodes[
      np.random.choice(len(episodes), size=n, replace=True)]
  
  def _query_entropy(self, n, *, scale):

    episodes = self._episodes()
    first = self._segment(episodes, size=self.size_segment)
    second = self._segment(episodes, size=self.size_segment)

    first, second = self._sample(first, n * scale), self._sample(second, n * scale)

    sa_t_1, sa_t_2, r_t_1, r_t_2 = self._compat(first, second)

    entropy, _ = self.get_entropy(sa_t_1, sa_t_2)
    
    top_k_index = (-entropy).argsort()[:self._effective_budget]
    r_t_1, sa_t_1 = r_t_1[top_k_index], sa_t_1[top_k_index]
    r_t_2, sa_t_2 = r_t_2[top_k_index], sa_t_2[top_k_index]
    
    # get labels
    sa_t_1, sa_t_2, r_t_1, r_t_2, labels = self.get_label(    
        sa_t_1, sa_t_2, r_t_1, r_t_2)
    
    if len(labels) > 0:
      self.put_queries(sa_t_1, sa_t_2, labels)
    
    return len(labels)

  def _query_entropy_aligned(self, 
                             n,
                             *,
                             cluster,
                             cluster_discard_outlier=True):

    episodes = self._episodes()
    first, second = self._every_aligned_pairs(
      episodes, self.size_segment, cluster, cluster_discard_outlier)
    
    sa_t_1, sa_t_2, r_t_1, r_t_2 = self._compat(first, second)

    entropy, _ = self.get_entropy(sa_t_1, sa_t_2)
    
    top_k_index = (-entropy).argsort()[:self._effective_budget]
    r_t_1, sa_t_1 = r_t_1[top_k_index], sa_t_1[top_k_index]
    r_t_2, sa_t_2 = r_t_2[top_k_index], sa_t_2[top_k_index]
    
    # get labels
    sa_t_1, sa_t_2, r_t_1, r_t_2, labels = self.get_label(    
        sa_t_1, sa_t_2, r_t_1, r_t_2)
    
    if len(labels) > 0:
      self.put_queries(sa_t_1, sa_t_2, labels)
    
    return len(labels)

  def _compute_clusters(self, 
                        episodes, 
                        cluster,
                        cluster_discard_outlier):
    u"""Cluster `episodes` based on their desired goals."""

    targets = []
    for episode in episodes:
      y = np.unique(episode.observation["desired_goal"], axis=0)
      assert len(y) == 1
      targets.append(y[0])
    targets = np.array(targets)

    cluster = cluster.fit(targets)

    labels_ = set(cluster.labels_)
    
    labels = []
    cluters = []
    for g in labels_:
      if cluster_discard_outlier and g == -1:
        continue
      mask = cluster.labels_ == g
      labels.append(g)
      cluters.append(episodes[mask])

    return labels, cluters
    targets = []
    for episode in episodes:
      target, = np.unique(
        episode.observation["desired_goal"], axis=0)
      targets.append(target)

    targets = np.array(targets)
    cluster = cluster.fit(targets)

    labels = cluster.labels_
    if labels.ndim != 1:
      raise AssertionError(f"{labels.ndim} != 1")

    index = np.argsort(labels)
    unique, (_, *sections) = np.unique(labels[index],
                                       return_index=True)
    return (unique.tolist(),
            np.split(episodes[index], sections))

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

    labels, clusters = self._compute_clusters(episodes, cluster, cluster_discard_outlier)
    pairs = []
    for label, cluster in zip(labels, clusters):
      pairs.append([*self._every_pairs(self._segment(cluster, length))])
    pairs = np.array(pairs, dtype=object)

    return (np.concatenate(pairs[:, 0], axis=0),
            np.concatenate(pairs[:, 1], axis=0))

  def _segment(self, episodes, size):
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
    next_index = self.buffer_index + total_sample
    if next_index >= self.capacity:
      self.buffer_full = True
      maximum_index = self.capacity - self.buffer_index
      np.copyto(self.buffer_seg1[self.buffer_index:self.capacity], sa_t_1[:maximum_index])
      np.copyto(self.buffer_seg2[self.buffer_index:self.capacity], sa_t_2[:maximum_index])
      np.copyto(self.buffer_label[self.buffer_index:self.capacity], labels[:maximum_index])

      remain = total_sample - (maximum_index)
      if remain > 0:
        np.copyto(self.buffer_seg1[0:remain], sa_t_1[maximum_index:])
        np.copyto(self.buffer_seg2[0:remain], sa_t_2[maximum_index:])
        np.copyto(self.buffer_label[0:remain], labels[maximum_index:])

      self.buffer_index = remain
    else:
      np.copyto(self.buffer_seg1[self.buffer_index:next_index], sa_t_1)
      np.copyto(self.buffer_seg2[self.buffer_index:next_index], sa_t_2)
      np.copyto(self.buffer_label[self.buffer_index:next_index], labels)
      self.buffer_index = next_index
          
  def get_label(self, sa_t_1, sa_t_2, r_t_1, r_t_2):
    assert len(sa_t_1) == self._effective_budget
    sum_r_t_1 = np.sum(r_t_1, axis=1)
    sum_r_t_2 = np.sum(r_t_2, axis=1)
    
    # skip the query
    if self.teacher_thres_skip > 0: 
      max_r_t = np.maximum(sum_r_t_1, sum_r_t_2)
      max_index = (max_r_t > self.teacher_thres_skip).reshape(-1)
      if sum(max_index) == 0:
        return None, None, None, None, []

      sa_t_1 = sa_t_1[max_index]
      sa_t_2 = sa_t_2[max_index]
      r_t_1 = r_t_1[max_index]
      r_t_2 = r_t_2[max_index]
      sum_r_t_1 = np.sum(r_t_1, axis=1)
      sum_r_t_2 = np.sum(r_t_2, axis=1)
    
    # equally preferable
    margin_index = (np.abs(sum_r_t_1 - sum_r_t_2) < self.teacher_thres_equal).reshape(-1)
    
    # perfectly rational
    seg_size = r_t_1.shape[1]
    temp_r_t_1 = r_t_1.copy()
    temp_r_t_2 = r_t_2.copy()
    for index in range(seg_size-1):
      temp_r_t_1[:,:index+1] *= self.teacher_gamma
      temp_r_t_2[:,:index+1] *= self.teacher_gamma
    sum_r_t_1 = np.sum(temp_r_t_1, axis=1)
    sum_r_t_2 = np.sum(temp_r_t_2, axis=1)
        
    rational_labels = 1*(sum_r_t_1 < sum_r_t_2)
    if self.teacher_beta > 0: # Bradley-Terry rational model
      r_hat = torch.cat([torch.Tensor(sum_r_t_1), 
                          torch.Tensor(sum_r_t_2)], axis=-1)
      r_hat = r_hat*self.teacher_beta
      ent = F.softmax(r_hat, dim=-1)[:, 1]
      labels = torch.bernoulli(ent).int().numpy().reshape(-1, 1)
    else:
      labels = rational_labels
    
    # making a mistake
    len_labels = labels.shape[0]
    rand_num = np.random.rand(len_labels)
    noise_index = rand_num <= self.teacher_eps_mistake
    labels[noise_index] = 1 - labels[noise_index]

    # equally preferable
    labels[margin_index] = -1 
    
    return sa_t_1, sa_t_2, r_t_1, r_t_2, labels

  def train_reward(self):
    ensemble_losses = [[] for _ in range(self.de)]
    ensemble_acc = np.array([0 for _ in range(self.de)])
    
    max_len = self.capacity if self.buffer_full else self.buffer_index
    total_batch_index = []
    for _ in range(self.de):
      total_batch_index.append(np.random.permutation(max_len))
    
    num_epochs = int(np.ceil(max_len/self.train_batch_size))
    total = 0
    
    for epoch in range(num_epochs):
      self._r_optimizer.zero_grad()
      loss = 0.0
      
      last_index = (epoch+1)*self.train_batch_size
      if last_index > max_len:
          last_index = max_len
          
      for member in range(self.de):
          
        # get random batch
        idxs = total_batch_index[member][epoch*self.train_batch_size:last_index]
        sa_t_1 = self.buffer_seg1[idxs]
        sa_t_2 = self.buffer_seg2[idxs]
        labels = self.buffer_label[idxs]
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
        curr_loss = self.CEloss(r_hat, labels)
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
    ensemble_losses = [[] for _ in range(self.de)]
    ensemble_acc = np.array([0 for _ in range(self.de)])
    
    max_len = self.capacity if self.buffer_full else self.buffer_index
    total_batch_index = []
    for _ in range(self.de):
      total_batch_index.append(np.random.permutation(max_len))
    
    num_epochs = int(np.ceil(max_len/self.train_batch_size))
    list_debug_loss1, list_debug_loss2 = [], []
    total = 0
    
    for epoch in range(num_epochs):
      self._r_optimizer.zero_grad()
      loss = 0.0
      
      last_index = (epoch+1)*self.train_batch_size
      if last_index > max_len:
        last_index = max_len
          
      for member in range(self.de):
          
        # get random batch
        idxs = total_batch_index[member][epoch*self.train_batch_size:last_index]
        sa_t_1 = self.buffer_seg1[idxs]
        sa_t_2 = self.buffer_seg2[idxs]
        labels = self.buffer_label[idxs]
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
        target_onehot = torch.zeros_like(r_hat).scatter(1, labels.unsqueeze(1), self.label_target)
        target_onehot += self.label_margin
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