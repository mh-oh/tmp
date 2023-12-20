

class AttrDict(dict):
  """
    Behaves like a dictionary but additionally has attribute-style access
    for both read and write.
    e.g. x["key"] and x.key are the same,
    e.g. can iterate using:  for k, v in x.items().
    Can sublcass for specific data classes; must call AttrDict's __init__().
    """
  def __init__(self, *args, **kwargs):
    dict.__init__(self, *args, **kwargs)
    self.__dict__ = self

  def copy(self):
    """
        Provides a "deep" copy of all unbroken chains of types AttrDict, but
        shallow copies otherwise, (e.g. numpy arrays are NOT copied).
        """
    return type(self)(**{k: v.copy() if isinstance(v, AttrDict) else v for k, v in self.items()})


class AnnotatedAttrDict(AttrDict):
  """
  This is an AttrDict that accepts tuples of length 2 as values, where the
  second element is an annotation.
  """
  def __init__(self, *args, **kwargs):
    argdict = dict(*args, **kwargs)
    valuedict = {}
    annotationdict = {}
    for k, va in argdict.items():
      if hasattr(va, '__len__') and len(va) == 2 and type(va[1]) == str:
        v, a = va
        valuedict[k] = v
        annotationdict[k] = a
      else:
        valuedict[k] = va
    super().__init__(self, **valuedict)
