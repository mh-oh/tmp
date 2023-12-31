
import time


def short_timestamp():
  """Returns string with timestamp"""
  import datetime
  return '{:%y%m%d:%H%M%S}'.format(datetime.datetime.now())


def return_elapsed_time(func):

  def wrap(*args, **kwargs):
    start = time.time()
    res = func(*args, **kwargs)
    elapsed = time.time() - start

    if res is None:
      return elapsed
    elif not isinstance(res, tuple):
      return res, elapsed
    else:
      return (*res, elapsed)

  return wrap