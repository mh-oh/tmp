
import wandb
import pandas as pd
import matplotlib.pyplot as plt


plt.rcParams.update(
  {"axes.titlesize": "medium",
   "axes.labelsize": "medium",
   "xtick.labelsize": "medium",
   "ytick.labelsize": "medium",
   "legend.fontsize": "medium",})


def isiterable(obj):
  try:
    iter(obj)
  except:
    return False
  else:
    return True


def join_outer(dfs, on):

  df, *dfs = dfs
  for other in dfs:
    df = df.set_index(on).join(other.set_index(on), how="outer").reset_index()
  return df


def assure_run(run):
  api = wandb.Api()
  if isinstance(run, str):
    run = api.run(f"rldev/experiments/{run}")
  return run


class Curve:

  def __init__(self, y, title=None, xmax=None, ymax=None):
    self.y = y
    self.title = title
  
  def draw(self, fig, ax, runs, labels=None):

    y, title = self.y, self.title

    if not isiterable(runs):
      runs = [runs]

    show_labels = True
    if labels is None:
      show_labels = False
      labels = range(len(runs))
    if not isiterable(labels):
      labels = [labels]

    if len(runs) != len(labels):
      raise ValueError()
    if show_labels:
      if len(set(labels)) != len(labels):
        raise ValueError()  

    dfs, x = [], "train/step"
    for run, label in zip(runs, labels):
      if label == x:
        raise ValueError()
      records = []
      for data in assure_run(run).scan_history():
        if data.get(y) is not None:
          records.append((data[x], data[y]))
      df = pd.DataFrame.from_records
      dfs.append(df(records, columns=[x, label]))
    df = join_outer(dfs, on=x)

    # Plot.
    for column in df.columns:
      if x != column:
        ax.plot(df.loc[:, x], df.loc[:, column], label=column)    
    if show_labels:
      ax.legend()
    if title is not None:
      ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    fig.tight_layout()
  
  def draw_reduce(self, fig, ax, runs, label=None, reduce="mean", err="std"):

    y, title = self.y, self.title

    if not isiterable(runs):
      runs = [runs]

    dfs, x = [], "train/step"
    if label == x:
      raise ValueError()
    for i, run in enumerate(runs):
      records = []
      for data in assure_run(run).scan_history():
        if data.get(y) is not None:
          records.append((data[x], data[y]))
      df = pd.DataFrame.from_records
      dfs.append(df(records, columns=[x, f"{label}.{i}"]))
    df = join_outer(dfs, on=x)

    if not reduce in {"mean", "median"}:
      raise ValueError()
    if not err in {"std", "sem"}:
      raise ValueError()

    def fn(df, name):
      return getattr(df, name)(axis=1)

    df = df.set_index(x)
    df[f"{label}.{reduce}"] = fn(df, reduce)
    df[f"{label}.{err}"] = fn(df, err)
    df = df.loc[:, [f"{label}.{reduce}", f"{label}.{err}"]]
    df = df.reset_index()

    # Plot.
    xdata = df.loc[:, x]
    ydata = df.loc[:, f"{label}.{reduce}"]
    e = df.loc[:, f"{label}.{err}"]
    ax.plot(xdata, ydata, label=label)
    ax.fill_between(xdata, ydata + e, ydata - e, alpha=0.4)

    if label is not None:
      ax.legend()
    if title is not None:
      ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    fig.tight_layout()


def fetch_runs(condition):
  api = wandb.Api()
  for run in api.runs("rldev/experiments"):
    if condition(run):
      yield run


if __name__ == "__main__":

  def condition(env, conf):
    def fn(run):
      config = run.config["_fields"]
      return ((config["env"] == env) and
              (config["conf"] == conf))
    return fn

  fig, ax = plt.subplots(figsize=(4,3))
  curve = Curve("test/success_rate", title="point-maze-medium")
  runs = fetch_runs(condition(env="point-maze-medium", conf="pebble"))
  curve.draw_reduce(fig, ax, runs, label="pebble")
  runs = fetch_runs(condition(env="point-maze-medium", conf="pebble-aligned"))
  curve.draw_reduce(fig, ax, runs, label="pebble-aligned")
  runs = fetch_runs(condition(env="point-maze-medium", conf="pebble-entropy"))
  curve.draw_reduce(fig, ax, runs, label="pebble-entropy")
  runs = fetch_runs(condition(env="point-maze-medium", conf="pebble-greedy-aligned-entropy-discard-outliers"))
  curve.draw_reduce(fig, ax, runs, label="pebble-entropy-aligned")
  fig.savefig("point-maze-medium.pdf")

  fig, ax = plt.subplots(figsize=(4,3))
  curve = Curve("test/success_rate", title="point-maze-o-3")
  runs = fetch_runs(condition(env="point-maze-o-3", conf="pebble"))
  curve.draw_reduce(fig, ax, runs, label="pebble")
  runs = fetch_runs(condition(env="point-maze-o-3", conf="pebble-aligned"))
  curve.draw_reduce(fig, ax, runs, label="pebble-aligned")
  fig.savefig("point-maze-o-3.pdf")