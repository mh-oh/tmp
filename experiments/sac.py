
import wandb

from rldev.buffers.basic import PEBBLEBuffer
from rldev.agents.pref.sac import SACPolicy
from rldev.agents.sac import SAC
from rldev.agents.pref import utils
from rldev.launcher import parse_args, push_args


# @configure("rldev.experiments")
def main():

  args = parse_args()
  if args.test_env is None:
    args.test_env = args.env
  
  from rldev.configs import get
  conf = push_args(get(args.conf), args)
  import subprocess, sys
  conf.cmd = sys.argv[0] + " " + subprocess.list2cmdline(sys.argv[1:])

  wandb.init(project="experiments",
             tags=conf.tag,
             entity="rldev",
             config=conf)

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

  def env_fn(name, seed):
    from rldev.environments import make
    def thunk():
      env = make(name); env.seed(seed); return env
    return thunk

  env = DummyVecEnv([env_fn(conf.env, conf.seed)])
  test_env = DummyVecEnv([env_fn(conf.test_env, conf.seed + 1234)])

  buffer = (
    lambda agent:
      PEBBLEBuffer(agent,
                   env.num_envs,
                   int(conf.replay_buffer_capacity),
                   env.observation_space,
                   env.action_space))

  observation_space = env.envs[0].observation_space
  action_space = env.envs[0].action_space
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

  agent = SAC(conf,
              env,
              test_env,
              policy,
              buffer)

  agent.run(conf.test_episodes)


if __name__ == "__main__":
  main()