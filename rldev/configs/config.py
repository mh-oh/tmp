

def push_args(config, args):
  for key, x in vars(args).items():
    if key in config:
      print(f"overrides {key}={x}")
    else:
      print(f"push new element {key}={x}")
    config[key] = x
  return config