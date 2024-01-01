
import gym
import gymnasium
import numpy as np
import time
import wandb

from abc import *
from collections import OrderedDict
from gym.spaces import Box, Dict
from overrides import overrides
from typing import Union, Callable, Optional

from rldev.utils import gym_types
from rldev.utils.structure import AttrDict
from rldev.utils.seeding import set_global_seeds
from rldev.utils.vec_env import SubprocVecEnv, DummyVecEnv

try:
  from baselines.common.atari_wrappers import make_atari, wrap_deepmind
except:
  pass


class EnvModule:
  """
  Used to wrap state-less environments in an mrl.Module.
  Vectorizes the environment.
  """
  def __init__(
      self,
      env: Union[str, Callable],
      num_envs: int = 1,
      seed: Optional[int] = None,
      name: Optional[str] = None,
      modalities = ['observation'],
      goal_modalities = ['desired_goal'],
      episode_life=True  # for Atari
  ):

    self.num_envs = num_envs

    if seed is None:
      seed = int(time.time())

    if isinstance(env, str):
      sample_env = make_env_by_id(env, seed, 0, episode_life)()
      env_list = [make_env_by_id(env, seed, i, episode_life) for i in range(num_envs)]
    else:
      sample_env = make_env(env, seed, 0)()
      env_list = [make_env(env, seed, i) for i in range(num_envs)]

    if num_envs == 1:
      self.env = DummyVecEnv(env_list)
    else:
      self.env = SubprocVecEnv(env_list)
    print('Initializing env!')

    self.render = self.env.render
    self.observation_space = sample_env.observation_space
    self.action_space = sample_env.action_space

    if isinstance(self.action_space, gym_types.Discrete):
      self.action_dim = self.action_space.n
      self.max_action = None
    else:
      assert isinstance(self.action_space, gym_types.Box), "Only Box/Discrete actions supported for now!"
      self.action_dim = self.action_space.shape[0]
      self.max_action = self.action_space.high[0]
      assert np.allclose(self.action_space.high,
                         -self.action_space.low), "Action high/lows must equal! Several modules rely on this"

    self.goal_env = False
    self.goal_dim = 0

    if isinstance(self.observation_space, gym_types.Dict):
      if goal_modalities[0] in self.observation_space.spaces:
        self.goal_env = True
        self.compute_reward = sample_env.compute_reward
        if hasattr(sample_env, 'achieved_goal'):
          self.achieved_goal = sample_env.achieved_goal
        for key in goal_modalities:
          assert key in self.env.observation_space.spaces
          self.goal_dim += int(np.prod(self.env.observation_space[key].shape))
      state_dim = 0
      for key in modalities:
        if key == 'desired_goal': continue
        assert key in self.env.observation_space.spaces
        state_dim += int(np.prod(self.env.observation_space[key].shape))
      self.state_dim = state_dim
    else:
      self.state_dim = int(np.prod(self.env.observation_space.shape))

    self.state = self.env.reset()

  def setup(self, agent):
    self._agent = agent

  def step(self, action):
    res = self.env.step(action)
    self.state = res[0]
    return res

  def reset(self, indices=None):
    if not indices:
      self.state = self.env.reset()
      return self.state
    else:
      reset_states = self.env.env_method('reset', indices=indices)
      if self.goal_env:
        for i, reset_state in zip(indices, reset_states):
          for key in reset_state:
            self.state[key][i] = reset_state[key]
      else:
        for i, reset_state in zip(indices, reset_states):
          self.state[i] = reset_state
      return self.state


def make_env(env_fn, seed, rank):
  """
  Utility function for multiprocessed env.
  
  :param env_id: (str) the environment ID
  :param num_env: (int) the number of environment you wish to have in subprocesses
  :param seed: (int) the inital seed for RNG
  """
  def _init():
    env = env_fn()
    env.seed(seed + rank)
    env = ReturnAndObsWrapper(env)
    return env

  set_global_seeds(seed)
  return _init


### BELOW is based on https://github.com/ShangtongZhang/DeepRL/blob/master/deep_rl/component/envs.py
### Added a fix for the VecEnv bug in infinite-horizon envs.


# adapted from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/envs.py
def make_env_by_id(env_id, seed, rank, episode_life=True):
  """Used for regular gym environments and Atari Envs"""
  def _init():
    if env_id.startswith("dm"):
      import dm_control2gym
      _, domain, task = env_id.split('-')
      env = dm_control2gym.make(domain_name=domain, task_name=task)
    else:
      env = gym.make(env_id)
    is_atari = hasattr(gym.envs, 'atari') and isinstance(env.unwrapped, gym.envs.atari.atari_env.AtariEnv)
    if is_atari:
      env = make_atari(env_id)
    env.seed(seed + rank)
    if is_atari:
      env = wrap_deepmind(env, episode_life=episode_life, clip_rewards=False, frame_stack=False, scale=False)
      obs_shape = env.observation_space.shape
      if len(obs_shape) == 3:
        env = TransposeImage(env)
      env = FrameStack(env, 4)
    env = ReturnAndObsWrapper(env)
    return env

  set_global_seeds(seed)
  return _init


class ReturnAndObsWrapper(gym.Wrapper):
  def __init__(self, env):
    gym.Wrapper.__init__(self, env)
    self.total_rewards = 0

  def step(self, action):
    obs, reward, done, info = self.env.step(action)
    info = AttrDict(info)
    self.total_rewards += reward
    if done:
      info.done_observation = obs
      info.terminal_state = True
      if info.get('TimeLimit.truncated'):
        info.terminal_state = False
      info.episodic_return = self.total_rewards
      self.total_rewards = 0
    else:
      info.terminal_state = False
      info.episodic_return = None
    return obs, reward, done, info

  def render(self, *args, **kwargs):
    return self.env.render(*args, **kwargs)

  def reset(self):
    return self.env.reset()

  def __getattr__(self, attr):
    return getattr(self.env, attr)


class FirstVisitDoneWrapper(gym.Wrapper):
  """A wrapper for sparse reward goal envs that makes them terminate
  upon achievement"""
  def step(self, action):
    obs, reward, done, info = self.env.step(action)
    if np.allclose(reward, 0.):
      done = True
      info['is_success'] = True
      if info.get('TimeLimit.truncated'):
        del info['TimeLimit.truncated']
    return obs, reward, done, info

  def reset(self):
    return self.env.reset()

  def __getattr__(self, attr):
    return getattr(self.env, attr)


class TransposeImage(gym.ObservationWrapper):
  def __init__(self, env=None):
    super(TransposeImage, self).__init__(env)
    obs_shape = self.observation_space.shape
    self.observation_space = gym.spaces.Box(self.observation_space.low[0, 0, 0],
                                            self.observation_space.high[0, 0, 0],
                                            [obs_shape[2], obs_shape[1], obs_shape[0]],
                                            dtype=self.observation_space.dtype)

  def observation(self, observation):
    return observation.transpose(2, 0, 1)


# The original LayzeFrames doesn't work well
class LazyFrames(object):
  def __init__(self, frames):
    """This object ensures that common frames between the observations are only stored once.
      It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
      buffers.
      This object should only be converted to numpy array before being passed to the model.
      You'd not believe how complex the previous solution was."""
    self._frames = frames

  def __array__(self, dtype=None):
    out = np.concatenate(self._frames, axis=0)
    if dtype is not None:
      out = out.astype(dtype)
    return out

  def __len__(self):
    return len(self.__array__())

  def __getitem__(self, i):
    return self.__array__()[i]


class FrameStack(gym.Wrapper):
  def __init__(self, env, k):
    """Stack k last frames.

    Returns lazy array, which is much more memory efficient.

    See Also
    --------
    baselines.common.atari_wrappers.LazyFrames
    """
    gym.Wrapper.__init__(self, env)
    self.k = k
    self.frames = deque([], maxlen=k)
    shp = env.observation_space.shape
    self.observation_space = spaces.Box(low=0, high=255, shape=(shp[0], shp[1], shp[2] * k), dtype=np.uint8)

  def reset(self):
    ob = self.env.reset()
    for _ in range(self.k):
      self.frames.append(ob)
    return self._get_ob()

  def step(self, action):
    ob, reward, done, info = self.env.step(action)
    self.frames.append(ob)
    return self._get_ob(), reward, done, info

  def _get_ob(self):
    assert len(self.frames) == self.k
    return LazyFrames(list(self.frames))


from gym import spaces
from rldev.utils.env import observation_spec, flatten_space, flatten_observation


class DictGoalEnv:

  def __init__(self, env):

    self._env = env
    self._dict_observation_space = env.observation_space
    self._dict_spec = observation_spec(self._dict_observation_space)

    self.observation_space = self._dict_observation_space
    for key in {"observation", "achieved_goal", "desired_goal"}:
      if key not in self.observation_space.spaces:
        raise ValueError()
    self.box_observation_space = flatten_space(self._dict_observation_space)

  def to_dict_observation(self, box_observation):

    if not isinstance(box_observation, np.ndarray):
      raise ValueError(f"")
    return spaces.unflatten(self._dict_observation_space, 
                            box_observation)

  def to_box_observation(self, dict_observation):    
    
    if not isinstance(dict_observation, (dict, OrderedDict)):
      raise ValueError(f"")
    return flatten_observation(self._dict_observation_space, 
                               dict_observation)

  def __getattr__(self, name):
    return getattr(self._env, name)


def create_env(name, seed, *args, **kwargs):

  from gym.envs import registry
  spec = registry.env_specs.get(name)
  if spec is not None:
    def make(name):
      return gym.make(name, *args, **kwargs)
  else:
    from gymnasium.envs import registry
    from gymnasium.wrappers.record_video import RecordVideo
    from rldev.environments.wrappers import GymApi
    spec = registry.get(name)
    if spec is not None:
      if "render_mode" not in kwargs:
        kwargs["render_mode"] = "rgb_array"
      def make(name):
        env = gymnasium.make(name, *args, **kwargs)
        # env = RecordVideo(env, video_folder=f"{wandb.run.dir}/videos")
        env = GymApi(env)
        env.seed(seed)
        return env
    else:
      raise NotImplementedError()

  return DictGoalEnv(make(name))


R = "r"
G = "g"
O_MAZE   = [[1, 1, 1, 1, 1, 1, 1],
            [1, G, 0, 0, 0, G, 1],
            [1, 0, 1, 1, 1, 0, 1],
            [1, R, 0, 0, 0, G, 1],
            [1, 1, 1, 1, 1, 1, 1]]
O_MAZE_1 = [[1, 1, 1, 1, 1, 1, 1],
            [1, G, 0, 0, 0, 0, 1],
            [1, 0, 1, 1, 1, 0, 1],
            [1, R, 0, 0, G, 0, 1],
            [1, 1, 1, 1, 1, 1, 1]]
O_MAZE_2 = [[1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, G, 0, 1],
            [1, 0, 1, 1, 1, G, 1],
            [1, R, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1]]
O_MAZE_3 = [[1, 1, 1, 1, 1, 1, 1],
            [1, G, 0, 0, 0, 0, 1],
            [1, 0, 1, 1, 1, G, 1],
            [1, R, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1]]

MEDIUM_MAZE = [[1, 1, 1, 1, 1, 1, 1, 1],
               [1, G, 0, 1, 0, 0, G, 1],
               [1, 0, 0, 1, 0, 1, 0, 1],
               [1, 1, R, 0, 0, 0, 1, 1],
               [1, 1, 1, 0, 1, 0, 0, 1],
               [1, 0, 0, 0, 0, 1, 0, 1],
               [1, 0, 0, 1, 0, 0, 0, 1],
               [1, 1, 1, 1, 1, 1, 1, 1]]

gym_envs = {"fetch-push": ("FetchPush-v2", (), {}),
            "fetch-push-dense": ("FetchPushDense-v2", (), {}),
            "fetch-reach": ("FetchReach-v2", (), {}),
            "fetch-reach-dense": ("FetchReachDense-v2", (), {}),
            "fetch-pick-and-place": ("FetchPickAndPlace-v2", (), {}),
            "fetch-pick-and-place-dense": ("FetchPickAndPlaceDense-v2", (), {}),
            "point-maze-u": ("PointMaze_UMaze-v3", (), {}),
            "point-maze-u-dense": ("PointMaze_UMazeDense-v3", (), {}),
            "point-maze-o": ("PointMaze_UMaze-v3", (), {"maze_map": O_MAZE, "render_mode": "rgb_array"}),
            "point-maze-o-dense": ("PointMaze_UMazeDense-v3", (), {"maze_map": O_MAZE}),
            "point-maze-o-1": ("PointMaze_UMaze-v3", (), {"maze_map": O_MAZE_1}),
            "point-maze-o-1-dense": ("PointMaze_UMazeDense-v3", (), {"maze_map": O_MAZE_1}),
            "point-maze-o-2": ("PointMaze_UMaze-v3", (), {"maze_map": O_MAZE_2}),
            "point-maze-o-2-dense": ("PointMaze_UMazeDense-v3", (), {"maze_map": O_MAZE_2}),
            "point-maze-o-3": ("PointMaze_UMaze-v3", (), {"maze_map": O_MAZE_3}),
            "point-maze-o-3-dense": ("PointMaze_UMazeDense-v3", (), {"maze_map": O_MAZE_3}),
            "point-maze-medium": ("PointMaze_UMaze-v3", (), {"maze_map": MEDIUM_MAZE}),
            "point-maze-medium-dense": ("PointMaze_UMazeDense-v3", (), {"maze_map": MEDIUM_MAZE}),}


def create_env_by_name(name, seed):
  try:
    name, args, kwargs = gym_envs.get(name)
  except:
    return create_metaworld_env(name, seed)
  else:
    return create_env(name, seed, *args, **kwargs)


class BoxGoalEnv:
  ...

  def __init__(self, env):

    self._env = env
    self._box_observation_space = env.observation_space
    self._box_spec = observation_spec(self._box_observation_space)

    def box(index):
      space = self._box_observation_space
      return Box(
        low=space.low[index], high=space.high[index], dtype=space.dtype)
    self._dict_observation_space = Dict(
      [(key, box(self.get_index(key))) for key in self.observation_keys])

  @property
  def observation_keys(self):
    return ["observation", "achieved_goal", "desired_goal"]

  @property
  def box_observation_space(self):
    return self._box_observation_space

  @property
  def dict_observation_space(self):
    return self._dict_observation_space

  @property
  def observation_space(self):
    return self.dict_observation_space

  def get_index(self, key):
    if key not in self.observation_keys:
      raise ValueError(
        f"key should be one of {self.observation_keys}")
    return self.index(key)

  @abstractmethod
  def index(self, key):
    ...
  
  def to_dict_observation(self, box_observation):

    if not isinstance(box_observation, np.ndarray):
      raise ValueError(f"")

    dict_observation = OrderedDict()
    for key in self.observation_keys:
      dict_observation[key] = box_observation[self.get_index(key)]

    return dict_observation

  def to_box_observation(self, dict_observation):    
    
    if not isinstance(dict_observation, (dict, OrderedDict)):
      raise ValueError(f"")

    spec = self._box_spec
    shapes = []
    for key in self.observation_keys:
      shapes.append(dict_observation[key].shape[:-len(spec.shape)])
    if len(set(shapes)) != 1:
      raise ValueError()
    box_observation = np.zeros(shapes[0] + spec.shape, dtype=spec.dtype)
    for key in self.observation_keys:
      box_observation[..., self.get_index(key)] = dict_observation[key]

    return box_observation

  def __getattr__(self, name):
    return getattr(self._env, name)

  def observation(self, box_observation):

    dict_observation = self.to_dict_observation(box_observation)
    dtype = self._box_spec.dtype
    assert (box_observation.astype(dtype) 
            != self.to_box_observation(dict_observation)).sum() <= 0
    return dict_observation

  def reset(self):
    return self.observation(self._env.reset())
  
  def step(self, action):
    box_observation, *extra = self._env.step(action)
    return self.observation(box_observation), *extra
  
  def compute_reward(self, achieved, desired, info):
    raise
    
    mode = self._reward_mode
    if mode == "dense":
      actions = info["action"]
      next_observations = self.to_box_observation(
        info["next_observation"])
      rewards = []
      for action, next_observation in zip(actions, next_observations):
        rewards.append(
          self._env.compute_reward(action, 
                                  next_observation)[0])
      return np.array(rewards)

    # Compute distance between goal and the achieved goal.
    d = goal_distance(achieved, desired)
    if mode == "sparse":
      return -(d > 0.05).astype(np.float32)
    elif mode == "distance":
      return -d
    
    raise ValueError(f"...")


class ButtonPressV2(BoxGoalEnv):

  @overrides
  def index(self, key):

    shape = self._box_observation_space.shape
    dim = np.prod(shape)
    if key == "desired_goal":
      return [36, 37, 38]
    elif key == "achieved_goal":
      return [4, 5, 6]
    elif key == "observation":
      return np.delete(
        np.arange(dim).reshape(shape), self.index("desired_goal"))


class ReachV2(BoxGoalEnv):

  @overrides
  def index(self, key):

    shape = self._box_observation_space.shape
    dim = np.prod(shape)
    if key == "desired_goal":
      return [36, 37, 38]
    elif key == "achieved_goal":
      return [4, 5, 6]
    elif key == "observation":
      return np.delete(
        np.arange(dim).reshape(shape), self.index("desired_goal"))


class PushV2(BoxGoalEnv):

  @overrides
  def index(self, key):

    shape = self._box_observation_space.shape
    dim = np.prod(shape)
    if key == "desired_goal":
      return [36, 37, 38]
    elif key == "achieved_goal":
      return [4, 5, 6]
    elif key == "observation":
      return np.delete(
        np.arange(dim).reshape(shape), self.index("desired_goal"))


from metaworld.envs.mujoco.env_dict import ALL_V2_ENVIRONMENTS as env_dict
metaworld_envs = {"button-press": (env_dict["button-press-v2"], ButtonPressV2),
                  "push": (env_dict["push-v2"], PushV2),
                  "pick-place": (env_dict["pick-place-v2"], None),
                  "reach": (env_dict["reach-v2"], ReachV2)}


def create_metaworld_env(name, seed):

  if name not in metaworld_envs:
    raise ValueError()

  cls, wrap = metaworld_envs[name]
  env = cls()
  env._freeze_rand_vec = False
  env._set_task_called = True
  env.seed(seed)

  from gym.wrappers.time_limit import TimeLimit
  from rldev.agents.core.bpref.rlkit.envs.wrappers import NormalizedBoxEnv  
  return wrap(TimeLimit(NormalizedBoxEnv(env), env.max_path_length))