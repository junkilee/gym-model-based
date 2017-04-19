import numpy as np
from sklearn.manifold import TSNE
import sys
import os
import tqdm
import pandas as pd

def read_data(filename, input_shape):
  data = pd.read_csv(filename, header=None).as_matrix()
  return data[:,:input_shape]

dqn_data = read_data('train_dqn.csv', 4)
pg_data = read_data('train_pg.csv', 4)

print(dqn_data.shape)
print(pg_data.shape)

whole = np.concatenate((dqn_data, pg_data), axis=0)
print(whole.shape)


X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
print(X.shape)
model = TSNE(n_components=2, random_state=0)
#np.set_printoptions(suppress=True)
output = model.fit_transform(whole[:30000]) 
print(output)
