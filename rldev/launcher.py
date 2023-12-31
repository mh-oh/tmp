
import argparse
import wandb

from tabulate import tabulate
from rldev.configs.registry import get


def push_args(conf, args):
  for key, x in vars(args).items():
    if key in conf:
      print(f"overrides {key}={x}")
    else:
      print(f"push new element {key}={x}")
    conf[key] = x
  return conf


def get_parser():

  parser = argparse.ArgumentParser()
  parser.add_argument("conf")
  parser.add_argument("--run", 
    required=True, type=str, help="name of this run")
  parser.add_argument("--seed",
    required=True, type=int, help="seed of this run")
  parser.add_argument("--env", 
    required=True, type=str, help="name of environment for training")
  parser.add_argument("--test_env", 
    default=None, type=str, help="name of environment for evaluation")
  parser.add_argument("--num_envs", 
    default=1, type=int, help="the number of envrionments for vectorization")
  parser.add_argument("--steps",
    default=1000000, type=int, help="training steps")
  parser.add_argument("--epoch_steps", 
    default=5000, type=int, help="length of an epoch in steps")

  return parser


def configure(project):

  parser = get_parser()
  args = parser.parse_args()
  if args.test_env is None:
    args.test_env = args.env

  conf = push_args(get(args.conf), args)
  import subprocess, sys
  conf.cmd = sys.argv[0] + " " + subprocess.list2cmdline(sys.argv[1:])

  def decorator(func):
    def wrap():
      api = wandb.Api()

      data = []
      for run in api.runs(project):
        data.append((run.name, run.id, run.state))
      print(tabulate(data,
                     headers=["run", "id", "state"]))

      names = {info[0] for info in data}
      while conf.run in names:
        conf.run = input(f"Name '{conf.run}' exists. try another name: ")

      wandb.init(project=project,
                 tags=[conf.run],
                 config=conf)

      return func(conf)
    return wrap
  return decorator




# @configure("rldev.experiments")
# def main(conf):
#   print(conf)
#   pass

# main()