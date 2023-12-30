
import copy


registry = {}

def get(name):
  try:
    return copy.deepcopy(registry[name])
  except:
    raise KeyError(f"unknown config '{name}'") from None

def register(name, conf):
  registry[name] = copy.deepcopy(conf)