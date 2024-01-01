
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
  parser.add_argument("--tag",
    nargs="+", type=str, help="tags of this run")
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


def parse_args():
  
  parser = get_parser()
  args = parser.parse_args()
  if args.tag is None:
    args.tag = []
  return args


def configure(project):

  args = parse_args()
  if args.test_env is None:
    args.test_env = args.env

  conf = push_args(get(args.conf), args)
  import subprocess, sys
  conf.cmd = sys.argv[0] + " " + subprocess.list2cmdline(sys.argv[1:])

  def decorator(func):
    def wrap():
      wandb.init(project=project,
                 entity="rldev",
                 tags=conf.tag,
                 config=conf)

      return func(conf)
    return wrap
  return decorator




# @configure("rldev.experiments")
# def main(conf):
#   print(conf)
#   pass

# main()