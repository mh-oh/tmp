
import numpy as np

from overrides import overrides

from rldev.agents.core.bpref import utils
from rldev.buffers.basic import PEBBLEBuffer
from rldev.agents.core.bpref.sac import DoubleQCritic, DiagGaussianActor, SACPolicy
from rldev.agents.core.bpref.reward_model import RewardModel
from rldev.agents.pebble import PEBBLE
from rldev.configs import Conf

config = Conf()

config.logging = {}
config.logging.wandb = "rldev"

config.experiment = 'PEBBLE'
config.segment = 50
config.activation = 'tanh'
config.num_seed_steps = 1000
config.num_unsup_steps = 9000
config.num_interact = 5000
config.reward_lr = 0.0003
config.reward_batch = 50
config.reward_update = 10
config.feed_type = 2
config.reset_update = 100
config.topK = 5
config.ensemble_size = 3
config.max_feedback = 10000
config.large_batch = 10
config.label_margin = 0.0
config.teacher_beta = -1
config.teacher_gamma = 1
config.teacher_eps_mistake = 0
config.teacher_eps_skip = 0
config.teacher_eps_equal = 0
config.reward_schedule = 0
config.steps = 1000000
config.replay_buffer_capacity = config.steps
config.test_every_n_steps = 10000
config.num_eval_episodes = 10
config.device = 'cuda'
config.log_every_n_steps = 3000
config.log_save_tb = True
config.save_video = False
config.seed = 2
config.env = 'push-v2'
config.gradient_update = 2
config.run = 'push.seed=2'
config.aligned_goals = False

config.policy = {}
config.policy.name = 'sac'
config.policy.cls = SACPolicy

config.policy.kwargs = {}
config.policy.kwargs.obs_dim = ...
config.policy.kwargs.action_dim = ...
config.policy.kwargs.action_range = ...
config.policy.kwargs.device = config.device
config.policy.kwargs.discount = 0.99
config.policy.kwargs.init_temperature = 0.1
config.policy.kwargs.alpha_lr = 0.0001
config.policy.kwargs.alpha_betas = [0.9, 0.999]
config.policy.kwargs.actor_lr = 0.0003
config.policy.kwargs.actor_betas = [0.9, 0.999]
config.policy.kwargs.actor_update_frequency = 1
config.policy.kwargs.critic_lr = 0.0003
config.policy.kwargs.critic_betas = [0.9, 0.999]
config.policy.kwargs.critic_tau = 0.005
config.policy.kwargs.critic_target_update_frequency = 2
config.policy.kwargs.batch_size = 512
config.policy.kwargs.learnable_temperature = True

config.critic = {}
config.critic.cls = DoubleQCritic
config.critic.kwargs = {}
config.critic.kwargs.hidden_dim = 256
config.critic.kwargs.hidden_depth = 3

config.actor = {}
config.actor.cls = DiagGaussianActor
config.actor.kwargs = {}
config.actor.kwargs.hidden_depth = 3
config.actor.kwargs.hidden_dim = 256
config.actor.kwargs.log_std_bounds = [-5, 2]


from experiments.metaworld.ddpg import BoxGoalEnv

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


def main(cfg):

  utils.set_seed_everywhere(cfg.seed)

  from rldev.utils.vec_env import DummyVecEnv as _DummyVecEnv
  class DummyVecEnv(_DummyVecEnv):

    def to_box_observation(self, observation):
      return self.envs[0].to_box_observation(observation)

    @property
    def _max_episode_steps(self):
      return self.envs[0]._max_episode_steps
  
  cls = {"button-press-v2": ButtonPressV2,
         "reach-v2": ReachV2,
         "push-v2": PushV2,
         }[cfg.env]
  env = DummyVecEnv([lambda: cls(utils.make_metaworld_env(cfg))])
  test_env = DummyVecEnv([lambda: cls(utils.make_metaworld_env(cfg))])

  buffer = (
    lambda agent:
      PEBBLEBuffer(agent,
                   env.num_envs,
                   int(cfg.replay_buffer_capacity),
                   env.observation_space,
                   env.action_space))

  cfg.policy.kwargs.obs_dim = env.envs[0].box_observation_space.shape[0]
  cfg.policy.kwargs.action_dim = env.action_space.shape[0]
  cfg.policy.kwargs.action_range = [
    float(env.action_space.low.min()), float(env.action_space.high.max())]
  
  cfg.critic.kwargs.obs_dim = cfg.policy.kwargs.obs_dim
  cfg.actor.kwargs.obs_dim = cfg.policy.kwargs.obs_dim
  cfg.critic.kwargs.action_dim = cfg.policy.kwargs.action_dim
  cfg.actor.kwargs.action_dim = cfg.policy.kwargs.action_dim
  cfg.policy.kwargs.critic_cfg = cfg.critic
  cfg.policy.kwargs.actor_cfg = cfg.actor

  policy = lambda agent: cfg.policy.cls(agent, **cfg.policy.kwargs)
  reward_model = (
    lambda agent:
      RewardModel(agent,
                  env.observation_space,
                  env.action_space,
                  env._max_episode_steps,
                  config.aligned_goals,
                  env.envs[0].box_observation_space.shape[0],
                  env.action_space.shape[0],
                  ensemble_size=cfg.ensemble_size,
                  size_segment=cfg.segment,
                  activation=cfg.activation, 
                  lr=cfg.reward_lr,
                  mb_size=cfg.reward_batch, 
                  large_batch=cfg.large_batch, 
                  label_margin=cfg.label_margin, 
                  teacher_beta=cfg.teacher_beta, 
                  teacher_gamma=cfg.teacher_gamma, 
                  teacher_eps_mistake=cfg.teacher_eps_mistake, 
                  teacher_eps_skip=cfg.teacher_eps_skip, 
                  teacher_eps_equal=cfg.teacher_eps_equal))

  agent = PEBBLE(cfg,
                 env,
                 test_env,
                 policy,
                 buffer,
                 reward_model)
  agent.run(cfg.num_eval_episodes)

if __name__ == '__main__':
  main(config)