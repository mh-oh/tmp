
from collections import Mapping


class Field:
  u"""A leaf node of a tree of variants."""

  def __init__(self, x, root, path=None, *, type=None, desc=None, check=None):

    self._x = x
    self._root = root
    self._path = path

    self._type = type
    self._desc = desc
    self._check = check

    if type is not None:
      if not isinstance(x, type):
        raise TypeError(type)

  def __repr__(self):
    return str(self._x)

  def get(self, expose=True):
    return self._x

  @property
  def path(self):
    return self._path


class Ref(Field):
  u""""""

  def __init__(self, x, root, path):
    super().__init__(x, root, path)

  def __repr__(self):
    return f"@`{self._x}`"

  def get(self, expose=True):
    if expose:
      return self._root.get(self._x, expose=expose)
    return self

  def set(self, value):
    ...


class ConstRef(Ref):
  u""""""

  def __init__(self, x, root, path):
    super().__init__(x, root, path)

  def set(self, value):
    raise TypeError("Cannot set value on const reference")


class Variant:

  _registry = {}

  @classmethod
  def list_variants(cls):
    return cls._registry.items()

  def register(self, name):
    u""""""

    reg = Variant._registry
    if name in reg:
      raise KeyError(f"variant name '{name}' conflicts")
    reg[name] = self

  def __init__(self, name=None, parent=None, path=""):
    
    object.__setattr__(self, "_fields", {})
    object.__setattr__(self, "_name", name)
    object.__setattr__(self, "_parent", parent)
    object.__setattr__(self, "_path", path)

    if name is not None:
      self.register(name)

  def __repr__(self):
    return "Variant({})".format(object.__getattribute__(self, "_fields"))

  @property
  def path(self):
    return self._path

  @property
  def root(self):
    for parent in self.upwards():
      pass
    return parent

  def upwards(self):
    parent = self
    while True:
      yield parent
      parent = object.__getattribute__(parent, "_parent")
      if parent is None:
        break

  def set(self, path, value, **kwargs):
    u""""""

    msg = "You cannot nest variant with '{}'"
    if isinstance(value, Variant):
      name = object.__getattribute__(self, "_name")
      if name is not None:
        raise ValueError(msg.format("name"))
      path = object.__getattribute__(self, "_path")
      if name is not None:
        raise ValueError(msg.format("path"))

    def set(obj, subpath, extra, value):
      key, *extra = extra
      fields = object.__getattribute__(obj, "_fields")
      subpath = (*subpath, key)
      sub = ".".join(subpath)
      if len(extra) > 0:
        if key not in fields:
          fields[key] = Variant(parent=obj, path=sub)
        set(fields[key], subpath, extra, value)
      else:
        # Silent initialization of sub variants.
        if isinstance(value, (Ref, ConstRef)):
          fields[key] = value
        elif isinstance(value, Mapping):
          value = Variant(parent=obj, path=sub)
        else:
          fields[key] = Field(value, self, path, **kwargs)

    set(self, (), path.split("."), value)

  # def __call__(self, path, value, **kwargs):
  #   self.set(path, value, **kwargs)

  def __setattr__(self, key, value):
    self.set(key, value)

  def __setitem__(self, path, value):
    self.set(path, value)

  def get(self, path, expose=False):
    u""""""

    field = self._find_field(path.split("."))
    return (field.get(expose) 
            if isinstance(field, Field) else field)

  def __getattr__(self, key):
    return self.get(key, expose=True)

  def __getitem__(self, path):
    return self.get(path, expose=True)

  def items(self):
    u""""""

    def items(obj, parents):
      fields = object.__getattribute__(obj, "_fields")
      for key, value in fields.items():
        if not isinstance(value, Variant):
          yield (*parents, key), value
        else:
          yield from items(value, (*parents, key))

    yield from items(self, ())

  def keys(self):
    u""""""

    def keys(obj, parents):
      fields = object.__getattribute__(obj, "_fields")
      for key, value in fields.items():
        if not isinstance(value, Variant):
          yield (*parents, key)
        else:
          yield from keys(value, (*parents, key))

    yield from keys(self, ())

  def values(self):
    u""""""

    def values(obj, parents):
      fields = object.__getattribute__(obj, "_fields")
      for key, value in fields.items():
        if not isinstance(value, Variant):
          yield value
        else:
          yield from values(value, (*parents, key))

    yield from values(self, ())

  def ref(self, path):
    u""""""
    
    if not isinstance(self, Variant):
      raise AssertionError()

    keys = path.split(".")
    target = self._find_field(keys)
    return Ref(target.path, 
               self.root, ".".join((self.path, *keys)))

  def const_ref(self, path):
    u""""""
    
    if not isinstance(self, Variant):
      raise AssertionError()

    keys = path.split(".")
    target = self._find_field(keys)
    return ConstRef(target.path, 
                    self.root, ".".join((self.path, *keys)))

  def _find_field(self, keys):

    key, *keys = keys
    field = object.__getattribute__(self, "_fields")[key]
    if len(keys) > 0:
      field = field._find_field(keys)
    return field


var = Variant(name="sac")
var.a = 1
var.set("b.c.d.e", 2, type=int)
var["c.d"] = 3
var.x = {}
var.y = var.b.c.d.ref("e")
var.z = var.ref("b.c.d.e")
print(var)
var.y = 1

for name, variant in Variant.list_variants():
  print(name)

# class Ref(FieldReference):
#   ...


# class _(Ref):
  
#   def __init__(self, type):
#     super().__init__(None, type, required=True)
  
#   def __repr__(self):
#     return f"? {self._field_type}"


# class Variant(ConfigDict):
  
#   def ref(self, key):
#     return super().get_ref(key)
  
#   def __init__(self):
#     super().__init__(type_safe=True, convert_dict=True)
  
#   def items(self, resolve=False):
    
#     if resolve:
#       return super().items(preserve_field_references=False)
    
#     def get(d, key):
#       try:
#         x = d[key]
#       except RequiredValueError:
#         x = d._fields[key]


# var = Variant()
# var.wrappers = []
# var.n_envs = 8
# var.seed = _(int)

# var.env = _(str)
# var.training_steps = 1_000_000
# var.lr = 1e-3
# var.learning_starts = 5_000
# var.batch_size = 256
# var.tau = 0.005
# var.gamma = 0.98
# var.train_every_n_steps = -1
# var.gradient_steps = 1
# var.logging_window = 100

# var.rb = {}
# var.rb.cls = _(str)
# var.rb.kwargs = {}
# var.rb.kwargs.capacity = var.ref("training_steps")

# var.qf = {}
# var.qf.cls = ...
# var.qf.kwargs = {}
# var.qf.kwargs.dims = [256, 256, 256]

# var.pi = {}
# var.pi.cls = ...
# var.pi.kwargs = {}
# var.pi.kwargs.dims = [256, 256, 256]

# var.test_env = _(str)
# var.test_every_n_episodes = 100
# var.test_episodes = 25


# # print(var.to_dict(preserve_field_references=False))
# def items(var):
#   u"""Nested iteration over `(key, value)` pairs.
#   `key` is a tuple of nested keys in the dictionary `d`."""

#   def f(d, parents):
#     for key in d.keys():
#       try:
#         x = d[key]
#       except RequiredValueError:
#         x = d._fields[key]
#       if not isinstance(x, ConfigDict):
#         yield (*parents, key), x
#       else:
#         yield from f(x, (*parents, key))

#   yield from f(var, ())


# for x in items(var):
#   print(x)


