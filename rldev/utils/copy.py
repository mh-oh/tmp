
import dill

def dillcopy(x):
  return dill.loads(dill.dumps(x))