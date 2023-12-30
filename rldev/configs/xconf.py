
from ml_collections.config_dict import ConfigDict, FieldReference


class Ref(FieldReference):
  ...

class Conf(ConfigDict):
  
  def ref(self, key):
    return super().get_ref(key)

def required(type):
  return Ref(None, type, required=True)

# conf = Conf()
# conf.x = 0
# conf.y = 1
# conf.z = 2
# conf.w = Conf()
# conf.w.x = 3
# conf.a = conf.w.ref("x")
# conf.w.x = 5
# conf.b = required(int)
# conf.b = 1
# print(conf.b)
