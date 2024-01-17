
import argparse
import itertools
import os
import subprocess

from gpustat import GPUStatCollection
from tabulate import tabulate
from tempfile import NamedTemporaryFile


u"""TMUX session."""

def new_tmux_sessions():

  tmp = NamedTemporaryFile()
  with open(tmp.name, "w") as fout:
    subprocess.run(["tmux", "ls"], stdout=fout)
    tmux_sessions = set()
    with open(tmp.name, "r") as fin:
      for line in fin:
        sess, _ = line.split(":", maxsplit=1)
        tmux_sessions.add(sess)

  numbers = []
  for sess in tmux_sessions:
    if sess.startswith("__autogen__"):
      numbers.append(int(sess[11:]))
  numbers.sort()
  if len(numbers) <= 0:
    numbers.append(0)

  for sess in itertools.count(numbers[-1] + 1):
    yield f"__autogen__{sess}" 


u"""Commands."""

def load_commands(file):
  commands = []
  with open(file, "r") as fin:
    for line in fin:
      line = line.rstrip()
      if line:
        commands.append(line)
  return commands


u"""GPU resources."""

def choose_gpus(n, me):

  gpu_stats = GPUStatCollection.new_query()

  host = gpu_stats.hostname
  print(f"We are in '{host}' with {len(gpu_stats.gpus)} gpus available.")
  print(f"Use 'gpustat' command to show the current usage.")
  print(f"")

  free_gpus = []
  for gpu in gpu_stats.gpus:
    gpu = gpu.jsonify()
    processes = gpu["processes"]
    users = []
    if len(processes) <= 0:
      free = True
    elif me is None:
      free = False
    else:
      free = True
      for process in processes:
        user = process["username"]; users.append(user)
        if user != me:
          free = False
    if free:
      free_gpus.append((gpu["index"], 
                        gpu["memory.total"] - gpu["memory.used"],
                        users))

  print(f"You can use the following gpus:")
  data = []
  for gpui, free_mem, users in free_gpus:
    if len(users) <= 0:
      data.append((f"{gpui}", f"{free_mem}", "empty"))
    else:
      for user in users:
        if user != me:
          raise AssertionError()
      data.append((f"{gpui}", f"{free_mem}", "you are using"))
  print(tabulate(data))
  print(f"")

  indicecs = set()
  for gpui, *_ in free_gpus:
    indicecs.add(int(gpui))

  while True:
    choices = input(f"Choose {n} gpu indices separated by ',': ")
    def validate(choices):
      try:
        choices = list(map(lambda token: int(token.strip()), 
                          choices.split(",")))
      except:
        return None, False
      if len(choices) != n:
        return None, False
      for index in choices:
        if not index in indicecs:
          return None, False
      return choices, True
    
    choices, success = validate(choices)
    if success:
      break
  print(f"You have chosen {choices}.")
  print(f"")

  return choices


u"""Launch the commands with TMUX sessions attached."""

CONTENTS = r"""
#!/bin/bash

{header}

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID={gpu}
export CUDA_VISIBLE_DEVICES={gpu}

{command}

rm {this}
"""

def run(commands, choices):

  if len(commands) != len(choices):
    raise AssertionError()

  print(f"Launching the commands...")
  how = zip(choices, commands, new_tmux_sessions())
  for gpui, cmd, sess in how:
    tmp = NamedTemporaryFile(delete=False)
    tmux_command = f'tmux new-session -d -s {sess} "bash {tmp.name}"'

    print(f"* {cmd}")
    print(f"  - gpu={gpui}, tmux-session='{sess}'")
    print(f"  - {tmux_command}")

    header = "source tools/launch/tmux_header.sh"
    with open(tmp.name, "w") as fout:
      fout.write(CONTENTS.format(
        header=header, gpu=gpui, command=cmd, this=tmp.name))
    os.system(tmux_command)


def main(file, me):

  commands = load_commands(file)
  print(f"You have requested {len(commands)} commands:")
  for cmd in commands:
    print(f"- {cmd}")
  print(f"")

  choices = choose_gpus(n=len(commands), me=me)
  run(commands, choices)


if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument("experiments", 
    help="Path to a file that contains commands for experiments")
  parser.add_argument("--me", 
    help="Your username in this local machine")
  args = parser.parse_args()

  main(args.experiments, args.me)