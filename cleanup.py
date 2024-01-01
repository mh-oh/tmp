
import argparse
import shutil
import wandb

from pathlib import Path
from tabulate import tabulate


def main(dir):

  while True:
    yesno = input(f"Are you sure you want to cleanup '{dir.resolve()}' [yes/no]? ")
    if yesno in {"yes", "no"}:
      break
  if yesno == "no":
    return

  api = wandb.Api()
  runs, data = set(), []
  for run in api.runs("rldev/experiments"):
    runs.add(run.id)
    data.append((run.id, run.state, run.user))
  print(tabulate(data))
  print()

  def local_run_paths():
    for path in dir.glob("run-*"):
      if not path.is_dir():
        continue
      _, _, runid = path.stem.split("-")
      if runid not in runs:
        yield path

  print("Removing...")  
  for path in local_run_paths():
    print(f"- {path}")
    shutil.rmtree(path)


if __name__ == "__main__":
  
  parser = argparse.ArgumentParser()
  parser.add_argument("wandb_dir", type=Path)
  args = parser.parse_args()
  main(args.wandb_dir)