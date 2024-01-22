
import re
from pathlib import Path


def find_rundir(runid):
  root = Path("wandb")
  for dir in root.glob("*"):
    match = re.match(
      r"run-(?P<date>\d+)_(?P<time>\d+)-(?P<runid>.*)", dir.stem)
    if match is not None:
      if runid == match.group("runid"):
        return dir
  return None


def run_info(run):
  fields = run.config["_fields"]
  return (fields["conf"],
          fields["seed"],
          fields["env"],
          fields["test_env"])

