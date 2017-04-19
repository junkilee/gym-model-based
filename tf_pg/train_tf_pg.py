'''
This code is based on:
https://github.com/hunkim/DeepRL-Agents
http://karpathy.github.io/2016/05/31/rl/
'''
import numpy as np
import tensorflow as tf
import gym
import sys, os
import gym_model_cartpole
from gym.envs.registration import register

if len(sys.argv) != 3:
    print("python train_tf_pg.py EPSILON ID!")
    sys.exit(1)

test_epsilon = sys.argv[1]
test_id = sys.argv[2]


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
      #'sstates_filename':'random.sstates.csv',
      'sstates_filename':'dqn.sstates.csv',
      'model_filename':'dqn.tmodel.h5',
      'epsilon': float(test_epsilon)
    }
)

print("test epsilon - {}".format(float(test_epsilon)))

#ENV_NAME_TRAIN = 'ModelBasedCartPoleFromDQNPolicy-v0'
#ENV_NAME_TRAIN = 'ModelBasedCartPoleFromDQNPolicy-v0'
ENV_NAME_TRAIN = 'ModelBasedCartPoleFromDQNPolicy-e-v0'
#ENV_NAME_TRAIN = 'ModelBasedCartPoleFromDQNPolicy-v0'
#env = gym.make('CartPole-v2')
#env = gym.make(ENV_NAME_TRAIN)
env = gym.make('CartPole-v2')
env_t = gym.make('CartPole-v2')

hidden_layer_neurons = 24
learning_rate = 1e-2

# Constants defining our neural network
input_size = env.observation_space.shape[0]
output_size = 1  # logistic regression, one p output

X = tf.placeholder(tf.float32, [None, input_size], name="input_x")

# First layer of weights
W1 = tf.get_variable("W1", shape=[input_size, hidden_layer_neurons],
                     initializer=tf.contrib.layers.xavier_initializer())
layer1 = tf.nn.relu(tf.matmul(X, W1))

# Second layer of weights
W2 = tf.get_variable("W2", shape=[hidden_layer_neurons, output_size],
                     initializer=tf.contrib.layers.xavier_initializer())
action_pred = tf.nn.sigmoid(tf.matmul(layer1, W2))

# Y (fake) and advantages (rewards)
Y = tf.placeholder(tf.float32, [None, output_size], name="input_y")
advantages = tf.placeholder(tf.float32, name="reward_signal")

# Loss function: log_likelihood * advantages
#log_lik = -tf.log(Y * action_pred + (1 - Y) * (1 - action_pred))     # using author(awjuliani)'s original cost function (maybe log_likelihood)
log_lik = -Y*tf.log(action_pred) - (1 - Y)*tf.log(1 - action_pred)    # using logistic regression cost function
loss = tf.reduce_sum(log_lik * advantages)

# Learning
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)


def discount_rewards(r, gamma=0.99):
    """Takes 1d float array of rewards and computes discounted reward
    e.g. f([1, 1, 1], 0.99) -> [1, 0.99, 0.9801] -> [1.22 -0.004 -1.22]
    http://karpathy.github.io/2016/05/31/rl/ """
    d_rewards = np.array([val * (gamma ** i) for i, val in enumerate(r)])

    # Normalize/standardize rewards
    d_rewards -= d_rewards.mean()
    d_rewards /= d_rewards.std()
    return d_rewards


# Setting up our environment
sess = tf.Session()
sess.run(tf.global_variables_initializer())

max_max_steps = 50000
max_num_episodes = 200

tstep = 0

f = open("train_pg0.csv", "w")
for step in range(max_num_episodes):
    # Initialize x stack, y stack, and rewards
    xs = np.empty(0).reshape(0, input_size)
    ys = np.empty(0).reshape(0, 1)
    rewards = np.empty(0).reshape(0, 1)

    reward_sum = 0
    observation = env.reset()

    while True:
        x = np.reshape(observation, [1, input_size])

        # Run the neural net to determine output
        action_prob = sess.run(action_pred, feed_dict={X: x})

        # Determine the output based on our net, allowing for some randomness
        action = 0 if action_prob < np.random.uniform() else 1

        # Append the observations and outputs for learning
        xs = np.vstack([xs, x])
        ys = np.vstack([ys, action])  # Fake action

        # Determine the outcome of our action
        pre_obs = observation
        observation, reward, done, _ = env.step(action)

        f.write("{},{},{},{}\n".format(','.join(map(str, pre_obs)), action, reward, done))
        f.flush()

        rewards = np.vstack([rewards, reward])
        reward_sum += reward

        if done:
            # Determine standardized rewards
            discounted_rewards = discount_rewards(rewards)
            l, _ = sess.run([loss, train],
                            feed_dict={X: xs, Y: ys, advantages: discounted_rewards})
            break

        if reward_sum >= 1000:
            print("Solved in {} episodes!".format(step))
            break

        tstep += 1

    # Print status
    print("Average reward for episode {}: {}. Loss: {}".format(step, reward_sum, l))

    if tstep >= max_max_steps:
        break
    if reward_sum >= 1000:
        break
f.close()

print("total steps : {}".format(tstep))
# See our trained bot in action
num_tests = 20

def run_test(env, num_tests):
    rewards = []
    reward_sum = 0
    observation = env.reset()
    for i in range(num_tests):
        #env.render()
        while True:
            x = np.reshape(observation, [1, input_size])
            action_prob = sess.run(action_pred, feed_dict={X: x})
            action = 0 if action_prob < 0.5 else 1  # No randomness
            pre_obs = observation
            observation, reward, done, _ = env.step(action)

            reward_sum += reward
            if done:
                print("Total score: {}".format(reward_sum))
                rewards.append(reward_sum)
                reward_sum = 0.0
                observation = env.reset()
                break
    return rewards

model_rewards = run_test(env, num_tests)
print("Model - Total mean reward: {} +- {}".format(np.mean(model_rewards), np.std(model_rewards) * 1.96 / np.sqrt(num_tests)))
real_rewards = run_test(env_t, num_tests)
print("Real - Total mean reward: {} +- {}".format(np.mean(real_rewards), np.std(real_rewards) * 1.96 / np.sqrt(num_tests)))


output_file = "outputs/output_pg_" + test_epsilon + "_" + test_id + ".txt"
f = open(output_file, 'w') 
f.write(",".join(map(str,model_rewards)) + "\n")
f.write(",".join(map(str,real_rewards)) + "\n")
f.close()
