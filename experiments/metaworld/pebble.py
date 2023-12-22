
import hydra

from rldev.agents.core.bpref import utils
from rldev.agents.core.bpref.replay_buffer import ReplayBuffer
from rldev.agents.core.bpref.reward_model import RewardModel
from rldev.agents.pebble import PEBBLE

@hydra.main(config_path='config/pebble.yaml', strict=True)
def main(cfg):

  utils.set_seed_everywhere(cfg.seed)

  env = utils.make_metaworld_env(cfg)
  test_env = utils.make_metaworld_env(cfg)

  # from stable_baselines3.common.vec_env import DummyVecEnv as _DummyVecEnv
  from rldev.utils.vec_env import DummyVecEnv as _DummyVecEnv
  class DummyVecEnv(_DummyVecEnv):
    
    # def reset(self):
    #   return super().reset()[0]
    
    # def step(self, action):
    #   obs, reward, done, info = super().step(action[np.newaxis, ...])
    #   return obs[0], reward[0], done[0], info[0]
    
    @property
    def _max_episode_steps(self):
      return self.envs[0]._max_episode_steps
  
  env = DummyVecEnv([lambda: env])
  test_env = DummyVecEnv([lambda: test_env])

  buffer = ReplayBuffer(
          env.observation_space.shape,
          env.action_space.shape,
          int(cfg.replay_buffer_capacity),
          "cuda")
  cfg.agent.params.obs_dim = env.observation_space.shape[0]
  cfg.agent.params.action_dim = env.action_space.shape[0]
  cfg.agent.params.action_range = [
      float(env.action_space.low.min()),
      float(env.action_space.high.max())
  ]
  policy = hydra.utils.instantiate(cfg.agent)
  reward_model = RewardModel(
      env.observation_space.shape[0],
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
      teacher_eps_equal=cfg.teacher_eps_equal)

  agent = PEBBLE(cfg,
                 env,
                 test_env,
                 lambda agent: policy,
                 buffer,
                 reward_model)
  agent.run(cfg.num_eval_episodes)

if __name__ == '__main__':
  main()