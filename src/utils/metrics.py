import numpy as np

def rmse(v,u):
    return np.sqrt(np.mean((u-v)**2))
