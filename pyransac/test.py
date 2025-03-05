import numpy as np
import sys
sys.path.append("./cmake-build-release")
import fitpoints
import dill

def save_cache_dill(obj, path):
   with open(path,'wb') as f:
      dill.dump(obj,f)

def load_cache_dill( path):
   with open(path, 'rb') as f:
      return dill.load(f)


if __name__=='__main__':
   v, n, l = load_cache_dill("./test_data_for_pyransac.pth")
   res = fitpoints.py_fit(v, n, 0.3, l)
   print(res)
