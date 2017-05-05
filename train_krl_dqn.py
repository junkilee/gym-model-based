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

if len(sys.argv) != 5:
    print("python train_krl_dqn.py PORTION EPSILON GAMMA ID!")
    sys.exit(1)

portion = sys.argv[1]
test_epsilon = sys.argv[2]
gamma = float(sys.argv[3])
test_id = sys.argv[4]

register(
    id='CartPole-v2',
    entry_point='gym.envs.classic_control:CartPoleEnv',
    max_episode_steps=1000,
    reward_threshold=1000.0,
)

register(
    id='ModelBasedCartPoleFromDQNPolicy-e-v0',
    entry_point='gym_model_cartpole.model:ModelBasedCartPoleEnv',
    kwargs={
      'name':'cartpolefromdqn',
      'sstates_filename':'model_000.sstates.csv',
      'model_filename':'model_000_' + portion  + '.h5',
      'epsilon': float(test_epsilon)
    }
)

register(
    id='ModelBasedCartPoleFromDQNPolicy-et-v0',
    entry_point='gym_model_cartpole.model:ModelBasedCartPoleEnv',
    kwargs={
      'name':'cartpolefromdqn',
      'sstates_filename':'model_000.sstates.csv',
      'model_filename':'model_000_' + portion  + '.h5',
      'epsilon': 0.0
    }
)

#ENV_NAME_TRAIN = 'ModelBasedCartPoleFromDQNPolicy-v0'
#ENV_NAME_TRAIN = 'ModelBasedCartPoleFromRandomPolicy-v0'
#ENV_NAME_TRAIN = 'ModelBasedCartPoleFromDQNPolicy-e0.6-v0'
ENV_NAME_TRAIN = 'ModelBasedCartPoleFromDQNPolicy-e-v0'
#ENV_NAME_TRAIN = 'CartPole-v2'
ENV_NAME_TEST2 = 'ModelBasedCartPoleFromDQNPolicy-et-v0'
ENV_NAME_TEST = 'CartPole-v2'

# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME_TRAIN)
env_test = gym.make(ENV_NAME_TEST)
env_test2 = gym.make(ENV_NAME_TEST2)
#np.random.seed(123)
#env.seed(123)
nb_actions = env.action_space.n
observation = env.reset()
print(observation)

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
dqn = DQNAgent(model=model, gamma=gamma, nb_actions=nb_actions, memory=memory, nb_steps_warmup=20,
               target_model_update=0.01, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

dqn.fit(env, nb_steps=50000, visualize=False, verbose=1)
state_errors = env.state_errors

#dqn.save_weights('dqn_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

nb_episodes = 20
result = dqn.test(env, nb_episodes=nb_episodes, visualize=False)
result2 = dqn.test(env_test, nb_episodes=nb_episodes, visualize=False)
result3 = dqn.test(env_test2, nb_episodes=nb_episodes, visualize=False)

#test_epsilon = sys.argv[1]
#test_id = sys.argv[2]
output_file = "outputs/output_dqn_" + portion + "_" + test_epsilon + "_" + str(gamma) + "_" + test_id + ".txt"
f = open(output_file, 'w') 
f.write(",".join(map(str,result.history['episode_reward'])) + "\n")
f.write(",".join(map(str,result3.history['episode_reward'])) + "\n")
f.write(",".join(map(str,result2.history['episode_reward'])) + "\n")
f.write(",".join(map(str,np.mean(state_errors, axis=0))) + "\n")
f.close()

print(result.epoch)
print(result3.epoch)
print(result2.epoch)
m_rew = result.history['episode_reward']
print("model {} +- {}".format(np.mean(m_rew), np.std(m_rew) * 1.96 / np.sqrt(10)))
r_rew = result3.history['episode_reward']
print("model_0.0 {} +- {}".format(np.mean(r_rew), np.std(r_rew) * 1.96 / np.sqrt(20)))
r_rew = result2.history['episode_reward']
print("real {} +- {}".format(np.mean(r_rew), np.std(r_rew) * 1.96 / np.sqrt(20)))
print(",".join(map(str,np.mean(state_errors, axis=0))))
