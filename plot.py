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


def curve(path, y, runs, labels=None, title=None, xmax=None):

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
      if data[y] is not None:
        records.append((data[x], data[y]))
    df = pd.DataFrame.from_records
    dfs.append(df(records, columns=[x, label]))
  df = join_outer(dfs, on=x)

  # Plot.
  figsize = (4, 3)
  fig, ax = plt.subplots(figsize=figsize)
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
  fig.savefig(path)


def curve_reduce(path, y, runs, label=None, title=None, reduce="mean", err="std", xmax=None):

  if not isiterable(runs):
    runs = [runs]

  dfs, x = [], "train/step"
  if label == x:
    raise ValueError()
  for i, run in enumerate(runs):
    records = []
    for data in assure_run(run).scan_history():
      if data[y] is not None:
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
  figsize = (4, 3)
  fig, ax = plt.subplots(figsize=figsize)

  xdata = df.loc[:, x]
  ydata = df.loc[:, f"{label}.{reduce}"]
  err = df.loc[:, f"{label}.{err}"]
  ax.plot(xdata, ydata, label=label)
  ax.fill_between(xdata, ydata + err, ydata - err, alpha=0.4)

  if label is not None:
    ax.legend()
  if title is not None:
    ax.set_title(title)
  ax.set_xlabel(x)
  ax.set_ylabel(y)
  fig.tight_layout()
  fig.savefig(path)

