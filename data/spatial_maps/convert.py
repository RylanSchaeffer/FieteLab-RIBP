import numpy as np
import joblib


data = dict(np.load('data.npz'))
for key, value in data.items():
    joblib.dump(value=value, filename=key+'.joblib')
