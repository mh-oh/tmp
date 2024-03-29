
import wandb

from rldev.buffers.basic import PEBBLEBuffer
from rldev.agents.pref.sac import SACPolicy
from rldev.agents.pref.reward_model import RewardModel
from rldev.agents.pebble import PEBBLE
from rldev.agents.pref import utils
from rldev.launcher import configure
from rldev.feature_extractor import Combine


@configure
def main(conf):

  utils.set_seed_everywhere(conf.seed)

  from rldev.utils.vec_env import DummyVecEnv as _DummyVecEnv
  class DummyVecEnv(_DummyVecEnv):

    @property
    def _max_episode_steps(self):
      try:
        return self.envs[0].spec.max_episode_steps
      except:
        try:
          return self.envs[0]._max_episode_steps
        except:
          return self.envs[0].max_episode_steps

  def env_fn(name, seed, wrappers):
    from rldev.environments import make
    def thunk():
      env = make(name, wrappers); env.seed(seed); return env
    return thunk

  env = DummyVecEnv([env_fn(conf.env, conf.seed, conf.env_wrappers)])
  test_env = DummyVecEnv([env_fn(conf.test_env, conf.seed + 1234, conf.env_wrappers)])

  buffer = (
    lambda agent:
      PEBBLEBuffer(agent,
                   env.num_envs,
                   int(conf.rb.capacity),
                   env.observation_space,
                   env.action_space,
                   disable_save=not conf.rb.save))

  observation_space = env.envs[0].observation_space
  action_space = env.envs[0].action_space
  feature_extractor = lambda agent: Combine(observation_space)
  policy = (
    lambda agent: 
      SACPolicy(agent,
                observation_space,
                action_space,
                conf.batch_size, 
                conf.discount,
                conf.tau, 
                conf.qf.cls,
                conf.qf.kwargs,
                conf.update_qf_target_every_n_steps,
                conf.pi.cls,
                conf.pi.kwargs,
                conf.update_pi_every_n_steps, 
                conf.learnable_alpha))

  reward_model = (
    lambda agent:
      RewardModel(agent,
                  env.observation_space,
                  env.action_space,
                  env._max_episode_steps,
                  conf.max_feedbacks,
                  conf.r.fusion,
                  conf.r.cls,
                  conf.r.kwargs,
                  budget=conf.budget, 
                  segment_length=conf.segment,
                  label_margin=conf.label_margin, 
                  teacher_beta=conf.teacher_beta, 
                  teacher_gamma=conf.teacher_gamma, 
                  teacher_eps_mistake=conf.teacher_eps_mistake, 
                  teacher_eps_skip=conf.teacher_eps_skip, 
                  teacher_eps_equal=conf.teacher_eps_equal))

  agent = PEBBLE(conf,
                 env,
                 test_env,
                 feature_extractor,
                 policy,
                 buffer,
                 reward_model)

  agent.run(conf.test_episodes)


if __name__ == "__main__":
  main()