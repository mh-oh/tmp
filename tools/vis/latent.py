
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pickle
import re
import torch as th
import torchextractor as thx
import wandb

from matplotlib.animation import FuncAnimation
from pathlib import Path

from rldev.agents.pebble import PEBBLE
from rldev.agents.pref.sac import SACPolicy
from rldev.agents.pref.reward_model import RewardModel
from rldev.agents.pref import utils
from rldev.buffers.basic import PEBBLEBuffer
from rldev.configs import get
from rldev.utils import torch as thu
from rldev.feature_extractor import Flatten


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
parser.add_argument("--n_episodes", type=int, default=3)
args = parser.parse_args()

save_dir = Path(f"tools/vis/outputs/{args.runid}")
save_dir.mkdir(parents=True, exist_ok=True)

api = wandb.Api()
run = api.run(f"rldev/experiments/{args.runid}")


def find_dir(runid):
  root = Path("/hdd/hdd1/omh/workspace/rldev/wandb")
  for dir in root.glob("*"):
    match = re.match(
      r"run-(?P<date>\d+)_(?P<time>\d+)-(?P<runid>.*)", dir.stem)
    if match is not None:
      if runid == match.group("runid"):
        return dir
  return None


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
                 env.action_space))


observation_space = env.envs[0].observation_space
action_space = env.envs[0].action_space
feature_extractor = lambda agent: Flatten(observation_space)
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


agent.load(find_dir(args.runid) / "files" / "agent")
r = agent._reward_model._r


_env = env.envs[0]

def draw_episode(episode):
  observation = _env.reset()

  observations, images, infos = [observation], [_env.render()], []
  r_hat, psi, phi = [], [], []
  done = False
  while not done:
    with utils.eval_mode(agent._policy):
      action = agent._policy.act(
        agent._feature_extractor(observation), sample=False)
    
    observation, reward, done, info = _env.step(action)
    observations.append(observation)
    images.append(_env.render())
    infos.append(info)

    def input(observation):
      return thu.torch(agent._feature_extractor(observation))

    _r = thx.Extractor(r._body[0], ["_psi", "_phi"])
    _r_hat, _features = _r(input(observation), thu.torch(action))
    r_hat.append(thu.numpy(_r_hat).squeeze())
    psi.append(thu.numpy(_features["_psi"]))
    phi.append(thu.numpy(_features["_phi"]))

  # First set up the figure, the axis, and the plot element we want to animate
  fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(3*2,2*2))
  # fig = plt.figure()
  # ax = plt.axes(xlim=(0, 2), ylim=(-2, 2))
  ax = axes[0, 0]
  im = ax.imshow(images[0])
  ax.set_xticks([])
  ax.set_yticks([])

  ax = axes[0, 1]
  ax.set_xlim(-1.5, 1.5)
  ax.set_ylim(-1.5, 1.5)
  psi_scatter = ax.scatter([], [], c="#D93226", label="\psi")
  phi_scatter = ax.scatter([], [], c="#78BA6B", label="\phi")
  psi_line, = ax.plot([], [], c="#D93226", alpha=0.5)
  phi_line, = ax.plot([], [], c="#78BA6B", alpha=0.5)
  # ax.legend()

  psi_norm = np.linalg.norm(np.array(psi), axis=-1)
  phi_norm = np.linalg.norm(np.array(phi), axis=-1)

  axes[1, 0].plot(psi_norm)
  axes[1, 0].set_title("L2 Norm of \psi")
  axes[1, 0].set_xlabel("Step")
  axes[1, 0].set_ylim(0, 1.5)
  axes[1, 1].plot(phi_norm)
  axes[1, 1].set_title("L2 Norm of \phi")
  axes[1, 1].set_xlabel("Step")
  axes[1, 1].set_ylim(0, 1.5)
  axes[1, 2].plot(np.array(r_hat))
  axes[1, 2].set_title("Predicted rewards")
  axes[1, 2].set_xlabel("Step")
  axes[0, 2].plot(np.array([info["sparse_reward"] for info in infos]))
  axes[0, 2].set_title("Success/Fail")
  axes[0, 2].set_xlabel("Step")

  def init():
    ...

  def update(t):

    im.set_data(images[t])

    psi_xy = np.array(psi[:t+1])
    psi_scatter.set_offsets(psi_xy[-1])
    psi_line.set_data(psi_xy[:, 0], psi_xy[:, 1])

    phi_xy = np.array(phi[:t+1])
    phi_scatter.set_offsets(phi_xy[-1])
    phi_line.set_data(phi_xy[:, 0], phi_xy[:, 1])

  anim = FuncAnimation(fig, update, 
                      frames=list(range(len(psi))), init_func=init)
  fig.tight_layout()
  anim.save(f"{str(save_dir)}/reward_episode_{episode}.mp4", fps=30, extra_args=['-vcodec', 'libx264'])


for episode in range(args.n_episodes):
  draw_episode(episode)