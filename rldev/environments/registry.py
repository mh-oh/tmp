
registry = set()

def get(name):
  try:
    return registry[name]
  except:
    raise KeyError(f"unknown config '{name}'") from None

def register(id, path, **kwargs):
  from gymnasium.envs.registration import register
  if id in registry:
    raise KeyError(f"'{id}' conflicts")
  registry.add(id)
  register(id, path, **kwargs)

def list_envs():
  for name in sorted(registry):
    yield name