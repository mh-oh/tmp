
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pickle
import re
import torch as th
import torchextractor as thx
import wandb

from pathlib import Path

from rldev.agents.pebble import PEBBLE
from rldev.agents.pref.sac import SACPolicy
from rldev.agents.pref.reward_model import RewardModel
from rldev.agents.pref import utils
from rldev.buffers.basic import PEBBLEBuffer
from rldev.configs import get
from rldev.utils import torch as thu
from rldev.feature_extractor import Combine

from tools.common import find_rundir


plt.rcParams.update(
  {"axes.titlesize": "medium",
   "axes.labelsize": "medium",
   "xtick.labelsize": "medium",
   "ytick.labelsize": "medium",
   "legend.fontsize": "medium",})


def run_info(run):
  fields = run.config["_fields"]
  return (fields["conf"],
          fields["seed"],
          fields["env"],
          fields["test_env"])


parser = argparse.ArgumentParser()
parser.add_argument("runid")
args = parser.parse_args()

save_dir = Path(f"tools/vis/outputs/{args.runid}")
save_dir.mkdir(parents=True, exist_ok=True)

api = wandb.Api()
run = api.run(f"rldev/experiments/{args.runid}")

conf, seed, env, test_env = run_info(run)
conf = get(conf)
conf.env = env
conf.test_env = test_env
conf.seed = seed


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
test_env = DummyVecEnv([env_fn(conf.env, conf.seed + 1234, conf.env_wrappers)])


buffer = (
  lambda agent:
    PEBBLEBuffer(agent,
                 env.num_envs,
                 int(conf.rb.capacity),
                 env.observation_space,
                 env.action_space,
                 disable_load=False))


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
                1,
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
               reward_model,
               logging=False)


agent.load(find_rundir(args.runid) / "files" / "agent")


agent._buffer._cursor = 1000000

def input(observation):
  return thu.torch(agent._feature_extractor(observation))

_r = agent._reward_model._r
_r = thx.Extractor(_r._body[0], ["_psi", "_phi"])
def forward(observation):
  return _r(input(observation), th.zeros(()))


_observations = agent._buffer._observations

_, features = forward(thu.torch(_observations))

psi = features["_psi"].detach().cpu().numpy()
n, n_envs, d = psi.shape
psi = psi.reshape((n * n_envs, d))


fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(2.6*1, 2*2))

ax = axes[0]
x = ax.scatter(psi[:, 0], psi[:, 1], alpha=0.1)
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_title("Discovered Targets")
c = fig.colorbar(x, ax=ax)
c.set_alpha(0.0)
c.set_ticks([])
c.solids.set_alpha(0.0)
c.outline.set_visible(False)

ax = axes[1]
psi = np.concatenate([psi, np.array([[-1.5, -1.5], [1.5, 1.5]])], axis=0)
h, xedges, yedges, image = ax.hist2d(psi[:, 0], psi[:, 1], bins=[25, 25], density=True)
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
cb = fig.colorbar(image, ax=ax)
cb.formatter.set_powerlimits((0, 0))
cb.ax.yaxis.set_offset_position("right")
cb.update_ticks()

fig.tight_layout()
fig.savefig(save_dir / "latent_targets.png")
print(save_dir / "latent_targets.png")

