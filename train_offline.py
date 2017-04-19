# vim: tabstop=2 softtabstop=2 shiftwidth=2 expandtab
import sys
import os
from datetime import datetime
from time import time
import numpy as np
import gym
import pandas as pd
from tqdm import tqdm
import network
from gym.envs.registration import register
from keras.models import Model

register(
    id='CartPole-v2',
    entry_point='gym.envs.classic_control:CartPoleEnv',
    max_episode_steps=1000,
    reward_threshold=1000.0,
)

def read_data(filename, input_shape):
  data = pd.read_csv(filename, header=None).as_matrix()
  data_list = []
  for i in tqdm(range(len(data)-1), desc="reading data", ascii=True):
    row = data[i]
    next_row = data[i+1]

    # x, x_dot, theta, theta_dot
    state_size = np.prod(input_shape)
    state = row[0:state_size].astype(float).reshape(input_shape)
    next_state = next_row[0:state_size].astype(float).reshape(input_shape)
    action = row[4]
    reward = row[5]
    termination = int(row[6])

    if termination is 0:
      data_list.append((state, next_state, action, reward, termination))
    else:
      data_list.append((state, np.zeros(input_shape), action, reward, termination))
  
  #action_translation = [-1, 1]

  data_size = len(data_list)
  print('data_size : ' + str(data_size))

  return data_list

def filter_data(data, portion):
  l = len(data)
  p = np.random.permutation(np.arange(l))[:int(l*portion)]
  data_list = []
  for i in p:
    data_list.append(data[i])
  return data_list

if __name__ == "__main__":
  if len(sys.argv) == 4:
    train_id = sys.argv[1]
    data_portion = float(sys.argv[2])
    test_id = sys.argv[3]
  else:
    print("python train_offline.py train_id data_portion test_id")
    sys.exit(1)

  # setting the directory and filename for train and test data files
  data_dir = "data"
  model_dir = "models"
  postfix = train_id
  train_filename = os.path.join(data_dir, 'train_' + postfix +'.csv')
  model_filename = os.path.join(model_dir, 'model_off_' + postfix)

  # default training parameters
  env_name = 'CartPole-v2'

  # setting up the environment
  env = gym.make(env_name)
  #env.seed(int(time()))

  freeze_batch = 16 # freeze the Q for n batches
  batch_size = 16
  test_turn = 1000 # every 4 epochs there is a real environemnt test
  num_test_episodes = 20

  state_shape = env.observation_space.shape
  num_acts = env.action_space.n
  num_units = 16
  num_hidden_layers = 3
  discount = 0.99

  epsilon = 0.05
  num_epochs = 2

  model, clone_model, trainable_model = network.get_dqn_model2(state_shape, num_acts, num_units, num_hidden_layers, discount)
  max_test = 0.0
  save_model = Model.from_config(model.get_config())
  save_model.set_weights(model.get_weights())

  train_data = read_data(train_filename, state_shape)
  train_data = filter_data(train_data, data_portion)

  train_size = len(train_data)

  total_losses = []
  print("train_size ", train_size, "data_portion", data_portion)
  for i in range(num_epochs * train_size):
    # prepare the batch
    states = np.zeros((batch_size,) + state_shape, dtype='float32')
    next_states = np.zeros((batch_size,) + state_shape, dtype='float32')
    actions = np.zeros((batch_size, ), dtype='int32')
    rewards = np.zeros((batch_size, ), dtype='float32')
    terminations = np.zeros((batch_size, ), dtype='int32')
    targets = np.zeros((batch_size, num_acts))

    for k in range(batch_size):
      #index = np.random.randint(0, max(batch_size, i % train_size))
      index = np.random.randint(0, train_size)
      states[k, :] = train_data[index][0]
      next_states[k, :] = train_data[index][1]
      actions[k] = train_data[index][2]
      rewards[k] = train_data[index][3]
      terminations[k] = train_data[index][4]

    # training phase
    next_q_vals = clone_model.predict_on_batch(next_states)
    q_batch = np.max(next_q_vals, axis=1).flatten()

    target_values = rewards + (1 - terminations.astype('float32')) * discount * q_batch

    targets = np.zeros((batch_size, num_acts), dtype='float32')
    dummy_targets = np.zeros((batch_size), dtype='float32')
    masks = np.zeros((batch_size, num_acts), dtype='float32')

    for k in range(batch_size):
      targets[k, actions[k]] = target_values[k]
      dummy_targets[k] = target_values[k]
      masks[k, actions[k]] = 1.

    loss = trainable_model.train_on_batch([states, targets, masks], [dummy_targets])
    total_losses.append(loss)

    gradual = False
    if not gradual:
      if i % 50:
        clone_model.set_weights(model.get_weights())
    else:
      weights = []
      tau = 1e-2
      for w1, w2 in zip(model.get_weights(), clone_model.get_weights()):
        weights += [w1 * tau + w2 * (1. - tau)]
      clone_model.set_weights(weights)

    # testing phase
    if i % 5000 == 0:
      total_losses = []
      total_rewards = []
      observation = env.reset()
      accum_reward = 0.0
      for j in range(num_test_episodes):
        while True:
          q_val = clone_model.predict_on_batch([observation.reshape((1, -1)).astype('float32')])
          if np.random.uniform(0,1) < epsilon:
            action = env.action_space.sample()
          else:
            action = np.argmax(q_val)
          observation, reward, done, _ = env.step(action)
          accum_reward += reward
          if done:
            total_rewards += [accum_reward]
            accum_reward = 0.0
            observation = env.reset()
            break
      print ('#epoch {}, #episodes {}, avg reward {} +- {}'.format(i, len(total_rewards), np.mean(total_rewards), np.std(total_rewards) * 1.96 / np.sqrt(len(total_rewards))))
      if np.mean(total_rewards) > max_test:
        save_model.set_weights(clone_model.get_weights())
        max_test = np.mean(total_rewards)


  # serialize model to JSON
  #model.save(model_filename + '.h5')
  #print("Saved model to disk")
  total_losses = []
  total_rewards = []
  observation = env.reset()
  accum_reward = 0.0
  for j in range(num_test_episodes):
    while True:
      q_val = save_model.predict_on_batch([observation.reshape((1, -1)).astype('float32')])
      if np.random.uniform(0,1) < epsilon:
        action = env.action_space.sample()
      else:
        action = np.argmax(q_val)
      observation, reward, done, _ = env.step(action)
      accum_reward += reward
      if done:
        total_rewards += [accum_reward]
        accum_reward = 0.0
        observation = env.reset()
        break
  print ('#epoch {}, #episodes {}, avg reward {} +- {}'.format(i, len(total_rewards), np.mean(total_rewards), np.std(total_rewards) * 1.96 / np.sqrt(len(total_rewards))))

output_file = "outputs/output_off_" + str(data_portion) + "_" + test_id + ".txt"
f = open(output_file, 'w') 
f.write(",".join(map(str,total_rewards)) + "\n")
f.close()
