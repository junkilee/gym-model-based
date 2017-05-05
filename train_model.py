# vim: tabstop=2 softtabstop=2 shiftwidth=2 expandtab
import sys
from datetime import datetime
from time import time
import os
import numpy as np
import gym
from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import l2, l1
import pandas as pd
from tqdm import tqdm

def get_transition_model():
  model = Sequential()
  # input (4 states + 1 action)

  model.add(Dense(20, input_dim=5, init='uniform', activation='relu'))
  model.add(Dense(20, init='uniform', activation='relu'))
  model.add(Dense(4, init='uniform', activation='linear'))
  # output difference

  model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy'])

  return model

def get_reward_model():
  model = Sequential()
  model.add(Dense(20, input_dim=5, init='uniform', activation='tanh'))
  model.add(Dense(20, init='uniform', activation='tanh'))
  model.add(Dense(20, init='uniform', activation='tanh'))
  model.add(Dense(4, init='uniform', activation='linear'))

  model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

  return model

def read_data(filename, is_display = False):
  data = pd.read_csv(filename, header=None).as_matrix()
  data_list = []
  for i in tqdm(range(len(data)-1), desc="reading data", ascii=True):
    row = data[i]
    next_row = data[i+1]

    # x, x_dot, theta, theta_dot
    state = row[0:4].astype(float)
    next_state = next_row[0:4].astype(float)
    next_diff = next_state - state

    action = row[4]
    reward = row[5]
    done = row[6]

    if is_display:
      #print('state', state)
      #print('next_state', next_state)
      #print('next_diff', next_diff)

      #raw_input('waiting...')
      pass

    if not done:
      data_list.append((state, next_diff, action))
  
  action_translation = [-1, 1]

  data_size = len(data_list)
  input_array = np.zeros([len(data_list), 5])
  output_array = np.zeros([len(data_list), 4])

  for i in tqdm(range(data_size), desc="creating numpy array", ascii=True):
    input_array[i][0:4] = data_list[i][0]
    input_array[i][4] = action_translation[data_list[i][2]]
    output_array[i][0:4] = data_list[i][1]

    if is_display:
      print (input_array[i])
      print (output_array[i])
      raw_input('waiting...')


  print('data_size : ' + str(data_size))

  return (input_array, output_array)

def filter_data(data, portion):
  l = len(data[0])
  p = np.random.permutation(np.arange(l))[:int(l*portion)]
  return (data[0][p], data[1][p])

if __name__ == "__main__":
  if len(sys.argv) == 3:
    train_id = sys.argv[1]
    data_portion = float(sys.argv[2])
  else:
    print("python train.py train_id data_portion")
    sys.exit(1)

  # setting the directory and filename for train and test data files
  data_dir = "data"
  model_dir = "model"
  postfix = train_id
  train_filename = os.path.join(data_dir, 'train_' + postfix +'.csv')
  test_filename = os.path.join(data_dir, 'train_' + postfix +'.csv')
  model_filename = os.path.join(model_dir, 'model_' + postfix + "_" + str(data_portion))

  # default training parameters
  epochs = 2
  batch_size = 16

  # initialize numpy
  #seed = 7
  #np.random.seed(seed)

  train_data = read_data(train_filename)
  #train_data = read_data(test_filename)
  train_data = filter_data(train_data, data_portion)
  test_data = read_data(test_filename)
  #test_data = read_data(train_filename)

  model = get_transition_model()

  print(len(train_data[0]))

  start_time = time()
  model.fit(train_data[0], train_data[1], nb_epoch = epochs, batch_size = batch_size, validation_data=(test_data[0], test_data[1]))
  end_time = time()
  print ('elasped time = {} secs'.format(end_time - start_time))
  score = model.evaluate(test_data[0], test_data[1], batch_size = batch_size)
  print('\n')
  print ('score = ', score)

  # serialize model to JSON
  model.save(model_filename + ".h5")
  #model_json = model.to_json()
  #with open(model_filename + ".json", "w") as json_file:
  #  json_file.write(model_json)
  # serialize weights to HDF5
  #model.save_weights(model_filename + ".weights.h5")
  print("Saved model to disk")
