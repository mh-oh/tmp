
from ml_collections.config_dict import ConfigDict, FieldReference


class Ref(FieldReference):
  ...

class Conf(ConfigDict):
  
  def ref(self, key):
    return super().get_ref(key)

def required(type):
  return Ref(None, type, required=True)

