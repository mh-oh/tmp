

def short_timestamp():
  """Returns string with timestamp"""
  import datetime
  return '{:%y%m%d:%H%M%S}'.format(datetime.datetime.now())