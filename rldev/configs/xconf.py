
import yaml
from ml_collections.config_dict import ConfigDict, FieldReference


class Ref(FieldReference):
  ...


class Conf(ConfigDict):
  
  def ref(self, key):
    return super().get_ref(key)


def required(type):
  return Ref(None, type, required=True)


def setdefault(conf, key, x):
  *keys, key = key.split(".")
  for k in keys:
    if k not in conf:
      conf[k] = Conf()
    conf = conf[k]
  conf[key] = x


def get(conf, key):
  x = conf
  for k in key.split("."):
    x = x[k]
  return x


def override(conf, key, x):
  
  *keys, k = key.split(".")
  if keys:
    conf = get(conf, ".".join(keys))
  if k not in conf:
    raise ValueError(f"unknown key '{key}'")
  conf[k] = x


def parse_overrides(conf, commands):

  def parse_cmd(cmd):
    key, x = map(str.strip, cmd.split("=", maxsplit=1))
    print(f"{key}, {x}, {type(x)}")
    return key, yaml.safe_load(x)
  
  for cmd in commands:
    override(conf, *parse_cmd(cmd))

