
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
  
  wandb.init(project="rldev.experiments",
             tags=[conf.run],
             config=conf)

  observation_space = env.envs[0].box_observation_space
  action_space = env.envs[0].action_space
  policy = (
    lambda agent: 
      SACPolicy(agent,
                      observation_space,
                      action_space,
                      conf.policy.kwargs.discount,
                      conf.policy.kwargs.init_temperature,
                      conf.policy.kwargs.alpha_lr, 
                      conf.policy.kwargs.alpha_betas,
                      conf.pi.kwargs.lr, 
                      conf.pi.kwargs.betas, 
                      conf.pi.kwargs.update_frequency, 
                      conf.qf.kwargs.lr,
                      conf.qf.kwargs.betas, 
                      conf.qf.kwargs.tau, 
                      conf.qf.kwargs.target_update_frequency,
                      conf.policy.kwargs.batch_size, 
                      conf.policy.kwargs.learnable_temperature,
                      conf.qf.kwargs.hidden_dim,
                      conf.qf.kwargs.hidden_depth,
                      conf.pi.kwargs.hidden_dim,
                      conf.pi.kwargs.hidden_depth,
                      conf.pi.kwargs.log_std_bounds,))

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