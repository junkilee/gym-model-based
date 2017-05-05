import numpy as np
import gym
from gym.envs.registration import register

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

import gym_model_cartpole
import sys, os


register(
    id='CartPole-v2',
    entry_point='gym.envs.classic_control:CartPoleEnv',
    max_episode_steps=1000,
    reward_threshold=1000.0,
)

ENV_NAME = 'CartPole-v2'

# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)

nb_actions = env.action_space.n
observation = env.reset()

# Next, we build a very simple model.
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())

memory = SequentialMemory(limit=20000, window_length=1)
policy = BoltzmannQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=20,
               target_model_update=0.01, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

dqn.fit(env, nb_steps=50000, visualize=False, verbose=1)

dqn.save_weights('dqn_weights.h5f', overwrite=True)

nb_episodes = 20
result = dqn.test(env, nb_episodes=nb_episodes, visualize=False)

print(result.epoch)
m_rew = result.history['episode_reward']
print("model {} +- {}".format(np.mean(m_rew), np.std(m_rew) * 1.96 / np.sqrt(10)))

dqn2 = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=20,
                target_model_update=0.01, policy=policy)
dqn2.compile(Adam(lr=1e-3), metrics=['mae'])
dqn2.load_weights('dqn_weights.h5f')

result = dqn2.test(env, nb_episodes=nb_episodes, visualize=False)
print(result.epoch)
m_rew = result.history['episode_reward']
print("model {} +- {}".format(np.mean(m_rew), np.std(m_rew) * 1.96 / np.sqrt(10)))
