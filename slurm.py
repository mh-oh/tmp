
import argparse
import datetime
import os
import yaml

from collections import defaultdict
from pathlib import Path
from tempfile import NamedTemporaryFile


def timestamp():
  now = datetime.datetime.now()
  return now.strftime('%y%m%d:%H%M%S')


u"""Commands."""

def load_commands(file):

  with open(file, "rb") as fin:
    data = yaml.safe_load(fin)
  
  cmds, n_cmds = defaultdict(list), 0
  for spec, group in data.items():
    partition, nodelist = spec.split("@")
    for cmd in group["commands"]:
      cmds[(partition, nodelist)].append(cmd)
      n_cmds += 1
  
  return cmds, n_cmds


u"""Launch the commands with slurm."""

CONTENTS = r"""#!/bin/bash

#SBATCH -J {job}
#SBATCH -o slurm.{job}.{time}.out
#SBATCH -t 3-00:00:00

#SBATCH -p {partition}
#SBATCH --gres=gpu:1

#SBATCH --nodelist={nodelist}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

srun -l /bin/hostname
srun -l /bin/pwd
srun -l /bin/date

{header}
{cmd}
"""

def run(cmds):

  print(f"Launching the commands...")
  for (partition, nodelist), group in cmds.items():
    print(f"* {partition} ({nodelist})")
    for cmd in group:
      tmp = NamedTemporaryFile(delete=False)
      slurm_command = f"sbatch {tmp.name}"
      kwargs = dict(job=Path(tmp.name).stem,
                    time=timestamp(),
                    partition=partition,
                    nodelist=nodelist,
                    header="source slurm_header.sh",
                    cmd=cmd)
      with open(tmp.name, "w") as fout:
        fout.write(CONTENTS.format(**kwargs))

      print(f"  - {cmd}")
      print(f"    ! {slurm_command}")
      os.system(slurm_command)


def main(file):

  cmds, n = load_commands(file)
  print(f"You have requested {n} commands:")

  for (partition, nodelist), group in cmds.items():
    print(f"* {partition} ({nodelist})")
    for cmd in group:
      print(f"  - {cmd}")
  print(f"")

  run(cmds)


if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument("experiments", 
    help="Path to a file that contains commands for experiments")
  args = parser.parse_args()

  main(args.experiments)