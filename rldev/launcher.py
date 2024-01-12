
import argparse
import click
import wandb

from tabulate import tabulate
from rldev.configs import get


# def push_args(conf, args):
#   for key, x in vars(args).items():
#     if key in conf:
#       print(f"overrides {key}={x}")
#     else:
#       print(f"push new element {key}={x}")
#     conf[key] = x
#   return conf


# def get_parser():

#   parser = argparse.ArgumentParser()
#   parser.add_argument("conf")
#   parser.add_argument("--tag",
#     nargs="+", type=str, help="tags of this run")
#   parser.add_argument("--seed",
#     required=True, type=int, help="seed of this run")
#   parser.add_argument("--env.key", 
#     required=True, type=str, help="name of environment for training")
#   parser.add_argument("--test_env", 
#     default=None, type=str, help="name of environment for evaluation")
#   parser.add_argument("--num_envs", 
#     default=1, type=int, help="the number of envrionments for vectorization")
#   parser.add_argument("--steps",
#     default=1000000, type=int, help="training steps")
#   parser.add_argument("--epoch_steps", 
#     default=5000, type=int, help="length of an epoch in steps")

#   return parser


# def parse_args():
  
#   parser = get_parser()
#   args = parser.parse_args()
#   if args.tag is None:
#     args.tag = []
#   return args


# def configure(project):

#   args = parse_args()
#   if args.test_env is None:
#     args.test_env = args.env

#   conf = push_args(get(args.conf), args)
#   import subprocess, sys
#   conf.cmd = sys.argv[0] + " " + subprocess.list2cmdline(sys.argv[1:])

#   def decorator(func):
#     def wrap():
#       wandb.init(project=project,
#                  entity="rldev",
#                  tags=conf.tag,
#                  config=conf)

#       return func(conf)
#     return wrap
#   return decorator


def configure(fun):

  @click.command()
  @click.argument("conf", type=str)
  @click.argument("overrides", nargs=-1, type=str)
  @click.option("-t", "--tag",
                type=str, multiple=True,
                help="""\b
                     Tag(s) of this run.
                     """)
  @click.option("--seed",
                metavar="INT",
                type=int, required=True,
                help="""\b
                     Random seed of this run.
                     """)
  @click.option("--env", 
                type=str, required=True,
                help="""\b
                     Name of environment for training.
                     Run 'list_envs.py' and choose one from the outputs.
                     """)
  @click.option("--test_env", 
                type=str,
                help="""\b
                     Name of environment for training.
                     Run 'list_envs.py' and choose one from the outputs.
                     If not specified, this use 'env'.
                     """)
  @click.option("--num_envs", 
                metavar="INT",
                type=int, default=1, show_default=True,
                help="""\b
                     Number of environments for vectorization.
                     This is used for both training and test environments.
                     """)
  @click.option("--epoch_steps", 
                metavar="INT",
                type=int, default=5000, 
                help="""\b
                     Length of an epoch in steps
                     """)
  def main(**kwargs):
    """RLDEV

    \b
    # CONF
    See 'README.md' for choosing an appropriate configuration.
    \b
    # OVERRIDES
    To override configuration keys, provide `<key>=<value>` 
    strings separated by whitespace(s) after -- (double-dash).
    For example,
      python experiments/*.py CONF [OPTIONS] -- foo.bar.baz=xyz
    """

    kwargs["test_env"] = kwargs["test_env"] or kwargs["env"]

    conf = get(kwargs["conf"])
    conf["conf"] = kwargs["conf"]
    conf["seed"] = kwargs["seed"]
    conf["env"] = kwargs["env"]
    conf["test_env"] = kwargs["test_env"]
    conf["num_envs"] = kwargs["num_envs"]
    conf["epoch_steps"] = kwargs["epoch_steps"]

    import subprocess, sys
    conf["cmd"] = (sys.argv[0] 
                   + " " + subprocess.list2cmdline(sys.argv[1:]))

    wandb.init(project="experiments",
               entity="rldev",
               tags=kwargs["tag"],
               config=conf)

    fun(conf)
  
  return main

