
from copy import deepcopy


registry = {}

def get(name):
  try:
    return deepcopy(registry[name])
  except:
    raise KeyError(f"unknown config '{name}'") from None

def register(name, conf):
  if name in registry:
    raise KeyError(f"'{name}' conflicts")
  registry[name] = deepcopy(conf)

def list_configs():
  for name in sorted(registry):
    yield name