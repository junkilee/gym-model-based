# vim: tabstop=2 softtabstop=2 shiftwidth=2 expandtab
import sys
from datetime import datetime
from time import time
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

def read_data(filename, is_display = False):
  data = pd.read_csv(filename, header=None).as_matrix()
  data_list = []

  f = open("start_states.csv", "w")
  check = True
  for i in tqdm(range(len(data)-1), desc="reading data", ascii=True):
    row = data[i]
    next_row = data[i+1]

    # x, x_dot, theta, theta_dot
    state = row[0:4].astype(float)

    if check:
      f.write("{}\n".format(','.join(map(str,state))))
      f.flush()
      check = False

    next_state = next_row[0:4].astype(float)
    next_diff = next_state - state

    action = row[4]
    reward = row[5]
    done = row[6]

    if done:
      check = True
  f.close()

read_data('train_01.csv')
