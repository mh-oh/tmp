
import numpy as np
import wandb

from collections import OrderedDict
from gym import spaces

from rldev.agents.core.bpref import utils
from rldev.buffers.basic import PEBBLEBuffer
from rldev.agents.core.bpref.sac import DoubleQCritic, DiagGaussianActor, SACPolicy
from rldev.agents.core.bpref.reward_model import RewardModel
from rldev.agents.pebble import PEBBLE
from rldev.configs.registry import get
from rldev.environments import create_env_by_name
from rldev.launcher import get_parser, push_args
from rldev.utils.env import observation_spec, flatten_space, flatten_observation


class DictGoalEnv:

  def __init__(self, env):

    self._env = env
    self._dict_observation_space = env.observation_space
    self._dict_spec = observation_spec(self._dict_observation_space)

    self.observation_space = self._dict_observation_space
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


# @configure("rldev.experiments")
def main():

  parser = get_parser()
  args = parser.parse_args()
  if args.test_env is None:
    args.test_env = args.env
  conf = push_args(get(args.conf), args)

  utils.set_seed_everywhere(conf.seed)

  from rldev.utils.vec_env import DummyVecEnv as _DummyVecEnv
  class DummyVecEnv(_DummyVecEnv):

    def to_box_observation(self, observation):
      return self.envs[0].to_box_observation(observation)

    @property
    def _max_episode_steps(self):
      return self.envs[0].spec.max_episode_steps

  env = DummyVecEnv([lambda: DictGoalEnv(create_env_by_name(conf.env))])
  test_env = DummyVecEnv([lambda: DictGoalEnv(create_env_by_name(conf.env))])

  buffer = (
    lambda agent:
      PEBBLEBuffer(agent,
                   env.num_envs,
                   int(conf.replay_buffer_capacity),
                   env.observation_space,
                   env.action_space))

  conf.policy.kwargs.obs_dim = env.envs[0].box_observation_space.shape[0]
  conf.policy.kwargs.action_dim = env.action_space.shape[0]
  conf.policy.kwargs.action_range = [
    float(env.action_space.low.min()), float(env.action_space.high.max())]
  
  conf.critic.kwargs.obs_dim = conf.policy.kwargs.obs_dim
  conf.actor.kwargs.obs_dim = conf.policy.kwargs.obs_dim
  conf.critic.kwargs.action_dim = conf.policy.kwargs.action_dim
  conf.actor.kwargs.action_dim = conf.policy.kwargs.action_dim
  conf.policy.kwargs.critic_cfg = conf.critic
  conf.policy.kwargs.actor_cfg = conf.actor

  wandb.init(project="rldev.experiments",
             tags=[conf.run],
             config=conf)

  policy = lambda agent: conf.policy.cls(agent, **conf.policy.kwargs)
  reward_model = (
    lambda agent:
      RewardModel(agent,
                  env.observation_space,
                  env.action_space,
                  env._max_episode_steps,
                  conf.aligned_goals,
                  env.envs[0].box_observation_space.shape[0],
                  env.action_space.shape[0],
                  ensemble_size=conf.ensemble_size,
                  size_segment=conf.segment,
                  activation=conf.activation, 
                  lr=conf.reward_lr,
                  mb_size=conf.reward_batch, 
                  large_batch=conf.large_batch, 
                  label_margin=conf.label_margin, 
                  teacher_beta=conf.teacher_beta, 
                  teacher_gamma=conf.teacher_gamma, 
                  teacher_eps_mistake=conf.teacher_eps_mistake, 
                  teacher_eps_skip=conf.teacher_eps_skip, 
                  teacher_eps_equal=conf.teacher_eps_equal))

  agent = PEBBLE(conf,
                 env,
                 test_env,
                 policy,
                 buffer,
                 reward_model)

  agent.run(conf.num_eval_episodes)


if __name__ == "__main__":
  main()