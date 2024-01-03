
registry = {}

def get(name):
  try:
    return registry[name]
  except:
    raise KeyError(f"unknown config '{name}'") from None

def register(env_fn, name):
  if name in registry:
    raise KeyError(f"'{name}' conflicts")
  registry[name] = env_fn