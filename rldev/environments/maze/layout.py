
from gymnasium_robotics.envs.maze.maps import R, G


registry = {}

def get(name):
  try:
    return registry[name]
  except:
    raise KeyError(f"unknown layout '{name}'") from None

def register(name, layout, target_pvals=None):
  if name in registry:
    raise KeyError(f"'{name}' conflicts")
  registry[name] = dict(layout=layout,
                        target_pvals=target_pvals)


register(
  "u-1-1",
  [[1, 1, 1, 1, 1],
   [1, G, 0, 0, 1],
   [1, 1, 1, 0, 1],
   [1, R, 0, 0, 1],
   [1, 1, 1, 1, 1]])

register(
  "u-1-2",
  [[1, 1, 1, 1, 1],
   [1, 0, 0, G, 1],
   [1, 1, 1, 0, 1],
   [1, R, 0, 0, 1],
   [1, 1, 1, 1, 1]])

register(
  "o-1-1",
  [[1, 1, 1, 1, 1, 1],
   [1, 0, 0, 0, G, 1],
   [1, 0, 1, 1, 0, 1],
   [1, R, 0, 0, 0, 1],
   [1, 1, 1, 1, 1, 1]])

register(
  "open-medium-2-1",
  [[1, 1, 1, 1, 1, 1, 1, 1],
   [1, G, 0, 0, 0, 0, 0, 1],
   [1, 0, 0, 0, 0, 0, 0, 1],
   [1, 0, 0, R, 0, 0, 0, 1],
   [1, 0, 0, 0, 0, 0, 0, 1],
   [1, 0, 0, 0, 0, 0, 0, 1],
   [1, 0, 0, 0, 0, 0, G, 1],
   [1, 1, 1, 1, 1, 1, 1, 1]])

register(
  "medium-1-1",
  [[1, 1, 1, 1, 1, 1, 1, 1],
   [1, 0, 0, 1, 0, 0, 0, 1],
   [1, 0, 0, 1, 0, 1, G, 1],
   [1, 1, 0, 0, 0, 0, 1, 1],
   [1, 1, 1, 0, 1, 0, 0, 1],
   [1, 0, 0, R, 0, 1, 0, 1],
   [1, 0, 0, 1, 0, 0, 0, 1],
   [1, 1, 1, 1, 1, 1, 1, 1]])

register(
  "medium-2-2",
  [[1, 1, 1, 1, 1, 1, 1, 1],
   [1, 0, 0, 1, 0, 0, 0, 1],
   [1, 0, 0, 1, 0, 1, G, 1],
   [1, 1, 0, 0, 0, 0, 1, 1],
   [1, 1, 1, 0, 1, 0, 0, 1],
   [1, G, 0, R, 0, 1, 0, 1],
   [1, 0, 0, 1, 0, 0, 0, 1],
   [1, 1, 1, 1, 1, 1, 1, 1]])

register(
  "medium-2-2-t28",
  [[1, 1, 1, 1, 1, 1, 1, 1],
   [1, 0, 0, 1, 0, 0, 0, 1],
   [1, 0, 0, 1, 0, 1, G, 1],
   [1, 1, 0, 0, 0, 0, 1, 1],
   [1, 1, 1, 0, 1, 0, 0, 1],
   [1, G, 0, R, 0, 1, 0, 1],
   [1, 0, 0, 1, 0, 0, 0, 1],
   [1, 1, 1, 1, 1, 1, 1, 1]], [0.2, 0.8])

register(
  "medium-3-1",
  [[1, 1, 1, 1, 1, 1, 1, 1],
   [1, G, 0, 1, 0, 0, G, 1],
   [1, 0, 0, 1, 0, 1, 0, 1],
   [1, 1, R, 0, 0, 0, 1, 1],
   [1, 1, 1, 0, 1, 0, 0, 1],
   [1, 0, 0, 0, 0, 1, 0, 1],
   [1, 0, 0, 1, 0, 0, G, 1],
   [1, 1, 1, 1, 1, 1, 1, 1]])

register(
  "medium-3-2",
  [[1, 1, 1, 1, 1, 1, 1, 1],
   [1, G, 0, 1, 0, 0, 0, 1],
   [1, 0, 0, 1, 0, 1, G, 1],
   [1, 1, 0, 0, 0, 0, 1, 1],
   [1, 1, 1, 0, 1, 0, 0, 1],
   [1, G, 0, R, 0, 1, 0, 1],
   [1, 0, 0, 1, 0, 0, 0, 1],
   [1, 1, 1, 1, 1, 1, 1, 1]])

register(
  "open-large-2-1",
  [[1, 1, 1, 1, 1, 1, 1, 1, 1],
   [1, G, 0, 0, 0, 0, 0, 0, 1],
   [1, 0, 0, 0, 0, 0, 0, 0, 1],
   [1, 0, 0, 0, 0, 0, 0, 0, 1],
   [1, 0, 0, 0, R, 0, 0, 0, 1],
   [1, 0, 0, 0, 0, 0, 0, 0, 1],
   [1, 0, 0, 0, 0, 0, 0, 0, 1],
   [1, 0, 0, 0, 0, 0, 0, 0, 1],
   [1, 0, 0, 0, 0, 0, 0, 0, 1],
   [1, 0, 0, 0, 0, 0, 0, 0, 1],
   [1, 0, 0, 0, 0, 0, 0, G, 1],
   [1, 1, 1, 1, 1, 1, 1, 1, 1]])

register(
  "large-2-1",
  [[1, 1, 1, 1, 1, 1, 1, 1, 1],
   [1, 0, 0, 0, 0, 0, 1, 0, 1],
   [1, 0, 1, 0, 1, 0, 1, G, 1],
   [1, 0, 0, 0, 1, 0, 0, 0, 1],
   [1, 0, 1, 1, 1, 0, 1, 1, 1],
   [1, 0, 0, 0, 0, 0, 0, 0, 1],
   [1, 1, 1, 0, 1, 1, 1, 0, 1],
   [1, 0, R, 0, 1, 0, 0, 0, 1],
   [1, G, 1, 0, 1, 1, 1, 1, 1],
   [1, 0, 1, 0, 1, 0, 0, 0, 1],
   [1, 0, 0, 0, 0, 0, 1, 0, 1],
   [1, 1, 1, 1, 1, 1, 1, 1, 1]])

register(
  "large-2-2",
  [[1, 1, 1, 1, 1, 1, 1, 1, 1],
   [1, 0, 0, 0, 0, 0, 1, 0, 1],
   [1, 0, 1, 0, 1, 0, 1, G, 1],
   [1, 0, 0, 0, 1, 0, 0, 0, 1],
   [1, 0, 1, 1, 1, 0, 1, 1, 1],
   [1, 0, 0, R, 0, 0, 0, 0, 1],
   [1, 1, 1, 0, 1, 1, 1, 0, 1],
   [1, 0, 0, 0, 1, 0, 0, 0, 1],
   [1, 0, 1, 0, 1, 1, 1, 1, 1],
   [1, 0, 1, 0, 1, 0, 0, 0, 1],
   [1, 0, G, 0, 0, 0, 1, 0, 1],
   [1, 1, 1, 1, 1, 1, 1, 1, 1]])

register(
  "large-2-3",
  [[1, 1, 1, 1, 1, 1, 1, 1, 1],
   [1, 0, 0, 0, 0, 0, 1, 0, 1],
   [1, 0, 1, 0, 1, 0, 1, 0, 1],
   [1, 0, 0, 0, 1, 0, 0, 0, 1],
   [1, G, 1, 1, 1, 0, 1, 1, 1],
   [1, 0, 0, R, 0, 0, 0, 0, 1],
   [1, 1, 1, 0, 1, 1, 1, 0, 1],
   [1, 0, 0, 0, 1, 0, 0, 0, 1],
   [1, 0, 1, 0, 1, 1, 1, 1, 1],
   [1, 0, 1, 0, 1, 0, 0, 0, 1],
   [1, 0, 0, 0, 0, 0, 1, G, 1],
   [1, 1, 1, 1, 1, 1, 1, 1, 1]])

register(
  "large-2-3-test",
  [[1, 1, 1, 1, 1, 1, 1, 1, 1],
   [1, 0, 0, 0, 0, 0, 1, 0, 1],
   [1, 0, 1, 0, 1, 0, 1, 0, 1],
   [1, 0, 0, 0, 1, 0, 0, 0, 1],
   [1, 0, 1, 1, 1, 0, 1, 1, 1],
   [1, 0, 0, R, 0, 0, 0, 0, 1],
   [1, 1, 1, 0, 1, 1, 1, 0, 1],
   [1, 0, 0, 0, 1, 0, 0, 0, 1],
   [1, 0, 1, 0, 1, 1, 1, 1, 1],
   [1, 0, 1, 0, 1, 0, 0, 0, 1],
   [1, 0, 0, 0, 0, 0, 1, G, 1],
   [1, 1, 1, 1, 1, 1, 1, 1, 1]])


register(
  "large-3-1",
  [[1, 1, 1, 1, 1, 1, 1, 1, 1],
   [1, 0, 0, 0, 0, 0, 1, 0, 1],
   [1, 0, 1, 0, 1, 0, 1, G, 1],
   [1, 0, 0, 0, 1, 0, 0, 0, 1],
   [1, 0, 1, 1, 1, 0, 1, 1, 1],
   [1, 0, 0, 0, 0, 0, 0, 0, 1],
   [1, 1, 1, 0, 1, 1, 1, 0, 1],
   [1, 0, R, 0, 1, 0, 0, 0, 1],
   [1, G, 1, 0, 1, 1, 1, 1, 1],
   [1, 0, 1, 0, 1, 0, 0, 0, 1],
   [1, 0, 0, 0, 0, G, 1, 0, 1],
   [1, 1, 1, 1, 1, 1, 1, 1, 1]])

register(
  "large-3-2",
  [[1, 1, 1, 1, 1, 1, 1, 1, 1],
   [1, 0, 0, 0, 0, 0, 1, 0, 1],
   [1, 0, 1, 0, 1, 0, 1, G, 1],
   [1, 0, 0, 0, 1, 0, 0, 0, 1],
   [1, G, 1, 1, 1, 0, 1, 1, 1],
   [1, 0, 0, R, 0, 0, 0, 0, 1],
   [1, 1, 1, 0, 1, 1, 1, 0, 1],
   [1, 0, 0, 0, 1, 0, 0, 0, 1],
   [1, 0, 1, 0, 1, 1, 1, 1, 1],
   [1, 0, 1, 0, 1, 0, 0, 0, 1],
   [1, 0, G, 0, 0, 0, 1, 0, 1],
   [1, 1, 1, 1, 1, 1, 1, 1, 1]])

register(
  "large-2-5",
  [[1, 1, 1, 1, 1, 1, 1, 1, 1],
   [1, G, 0, 0, 0, 0, 0, 0, 1],
   [1, 0, 0, 0, 0, 0, 0, 0, 1],
   [1, 0, 0, 0, 0, 0, 0, 0, 1],
   [1, 0, 0, 0, R, 0, 0, 0, 1],
   [1, 0, 0, 0, 0, 0, 0, 0, 1],
   [1, 0, 0, 0, 0, 0, 0, 0, 1],
   [1, 0, 0, 0, 0, 0, 0, 0, 1],
   [1, 0, 0, 0, 0, 0, 0, 0, 1],
   [1, 0, 0, 0, 0, 0, 0, 0, 1],
   [1, 0, 0, 0, 0, 0, 0, G, 1],
   [1, 1, 1, 1, 1, 1, 1, 1, 1]])

