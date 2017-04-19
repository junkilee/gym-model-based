# vim: tabstop=2 softtabstop=2 shiftwidth=2 expandtab
import sys
from datetime import datetime
from time import time
import os
import numpy as np
import gym
import gym.spaces
from gym.utils import seeding
from gym.envs.registration import register
from keras.models import load_model
import pandas as pd
from tqdm import tqdm
import math
import copy

def read_start_states(filename):
  data = pd.read_csv(filename, header=None).as_matrix()
  data_list = []
  for i in tqdm(range(len(data)-1), desc="reading data", ascii=True):
    row = data[i]
    data_list.append(row)

  data_size = len(data_list)
  input_array = np.zeros([len(data_list), 4])

  for i in tqdm(range(data_size), desc="creating numpy array", ascii=True):
    input_array[i][0:4] = data_list[i]

  print('data_size : ' + str(data_size))

  return input_array

class ModelBasedCartPoleEnv(gym.Env):
  def __init__(self, name='cartpole',
      model_dir = 'models',
      sstates_filename = 'cartpole.sstates.csv', 
      model_filename = 'cartpole.tmodel.h5',
      max_episode_steps = 1000,
      epsilon = 1.0,
      sstate_mode = 'data',
      seed = None):

    self.name = name
    self.nb_actions = 2
    self.start_states = read_start_states(os.path.join(model_dir, sstates_filename))
    self.sstate_mode = sstate_mode
    self.action_space = gym.spaces.Discrete(self.nb_actions)
    self.model = load_model(os.path.join(model_dir, model_filename))
    self.num_steps = 0
    self.max_episode_steps = max_episode_steps
    self.current_state = False
    self.action_translation = [-1, 1]
    self.epsilon = epsilon

    self.theta_threshold_radians = 12 * 2 * math.pi / 360
    self.x_threshold = 2.4

    high = np.array([self.x_threshold * 2,
                     np.finfo(np.float32).max,
                     self.theta_threshold_radians * 2,
                     np.finfo(np.float32).max])
    
    self.observation_space = gym.spaces.Box(-high, high)

    self._seed(seed)

    print('# of start states', len(self.start_states))

    if self.sstate_mode == 'env':
      self.gym_env = gym.make("CartPole-v0")

  def _seed(self, seed=None):
    self.np_random, seed1 = seeding.np_random(seed)
    # Derive a random seed. This gets passed as a uint, but gets
    # checked as an int elsewhere, so we need to keep it below
    # 2**31.
    seed2 = seeding.hash_seed(seed1 + 1) % 2**31
    return [seed1, seed2]

  def _reset(self):
    if self.sstate_mode == 'env':
      self.current_state = self.gym_env.reset()
      self.num_steps = 0
      return self.current_state
    else:
      index = self.np_random.randint(len(self.start_states))
      self.current_state = copy.deepcopy(self.start_states[index])
      self.num_steps = 0
      return self.current_state

  def _step(self, action):
    input = np.zeros(5)
    input[0:4] = self.current_state

    randval = self.np_random.uniform()

    c_action = action
    if randval <= self.epsilon:
      c_action = self.np_random.randint(self.nb_actions)

    input[4] = self.action_translation[c_action]

    output = self.model.predict(np.expand_dims(input, 0), batch_size = 1)

    next_state = output[0] + input[0:4]
    
    x, x_dot, theta, theta_dot = next_state
    self.num_steps += 1

    done =  x < -self.x_threshold \
      or x > self.x_threshold \
      or theta < -self.theta_threshold_radians \
      or theta > self.theta_threshold_radians 

    reward = 1
    if done:
      reward = 0

    if self.num_steps >= self.max_episode_steps:
      done = True

    self.current_state = next_state
    return self.current_state, reward, done, {}

if __name__ == '__main__':

  register(
      id='ModelBasedCartPoleFromRandomPolicy-v0',
      entry_point='model:ModelBasedCartPoleEnv',
      kwargs={
        'name':'cartpolefromrandom',
        'sstates_filename':'random.sstates.csv',
        'model_filename':'random.tmodel.h5'
      }
  )

  register(
      id='ModelBasedCartPoleFromDQNPolicy-v0',
      entry_point='model:ModelBasedCartPoleEnv',
      kwargs={
        'name':'cartpolefromdqn',
        'sstates_filename':'dqn.sstates.csv',
        'model_filename':'dqn.tmodel.h5'
      }
  )

  env = gym.make('ModelBasedCartPoleFromDQNPolicy-v0')
  observation = env.reset()

  for i in range(30):
    new_observation, reward, done, _ = env.step(np.random.randint(2))
    print (observation, new_observation, reward, done)
    if done:
      break
  
  env = gym.make('ModelBasedCartPoleFromRandomPolicy-v0')
  observation = env.reset()

  for i in range(30):
    new_observation, reward, done, _ = env.step(np.random.randint(2))
    print (observation, new_observation, reward, done)
    if done:
      break
