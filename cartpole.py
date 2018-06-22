import tensorflow as tf
from tensorflow.contrib import layers
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.io as sio
import time
import argparse
import gym
import argparse
from copy import deepcopy
from datetime import datetime


alpha = 2e-4
epsilon = 0.15

gamma = 1.0
dim_state = 4

max_episodes = 15000
num_trials = 100
buffer_size = 10000

def OU_next(y_prev, mu=0, sig=1, th=0.15):
    y_next = y_prev + th * (mu - y_prev) + sig * np.sqrt(2) * np.sqrt(th) * np.random.normal()
    return y_next

class NN():
    def __init__(self):

        self.sess = tf.Session()

        self.state_ph = tf.placeholder(tf.float32, [None, dim_state])
        self.q_ph = tf.placeholder(tf.float32, [None, 1])
        self.action_ph = tf.placeholder(tf.int32, [None,])

        self.w1 = tf.Variable(tf.truncated_normal([dim_state, 64], 0, 0.1))
        self.b1 = tf.Variable(tf.zeros([64]))

        self.h1 = tf.tanh(tf.matmul(self.state_ph, self.w1) + self.b1)

        self.w2 = tf.Variable(tf.truncated_normal([64, 2], 0, 0.1))
        self.b2 = tf.Variable(tf.zeros([2]))

        self.q = tf.matmul(self.h1, self.w2) + self.b2

        self.v_a = tf.reshape(tf.one_hot(self.action_ph, depth=2), [-1, 2])

        q = tf.reshape(self.q, [-1, 2])

        self.q_a = tf.reshape(tf.reduce_sum(q*self.v_a, axis=1), [-1, 1])

        self.lose = tf.losses.mean_squared_error(self.q_ph, self.q_a)

        self.optimizer = tf.train.AdamOptimizer(alpha)
        self.train_step = self.optimizer.minimize(self.lose)

        self.sess.run(tf.global_variables_initializer())

    def __call__(self, s):
        s = np.reshape(s, [1, dim_state])
        q_out = self.sess.run(self.q, feed_dict={self.state_ph:s})
        return np.reshape(q_out, [2])

    def train(self, s, a, q_target):
        s = np.reshape(s, [-1, dim_state])
        a = np.reshape(a, [-1])
        q_target = np.reshape(q_target, [-1, 1])

        self.sess.run(self.train_step, feed_dict={self.state_ph:s, self.action_ph:a, self.q_ph:q_target})


########################## white noise #################################



env = gym.make('CartPole-v1')

PC = []
GS = []
ST = []
for trial in range(num_trials):

    nn = NN()

    batch_q = np.array([[0]])
    batch_s = np.array([[0,0,0,0]])
    batch_a = np.array([0])

    performance_curve = []
    global_steps = []
    global_step = 0
    steps = []

    for episode in range(max_episodes):

        state = env.reset()
        done = 0
        step = 0
        total_reward = 0

        while not done and step < 200:

            step += 1
            global_step += 1

            tmp = np.random.normal(0, 1)

            if abs(tmp) < epsilon: # epsilon-greedy
                action = np.random.choice(2)
            else:
                action = np.argmax(nn(state))

            new_state, reward, done, _ = env.step(action)
            total_reward += reward

            if not done:
                Q_target = reward + gamma * np.max(nn(new_state))
            else:
                Q_target = reward

            batch_s = np.append(batch_s, np.reshape(state,[1,dim_state]), axis=0)
            batch_a = np.append(batch_a, action)
            batch_q = np.append(batch_q, np.reshape(Q_target, [1,1]), axis=0)

            if global_step > buffer_size + buffer_size / 5 and global_step % (buffer_size/5) == 0:
                batch_s = batch_s[int(buffer_size / 5):]
                batch_a = batch_a[int(buffer_size / 5):]
                batch_q = batch_q[int(buffer_size / 5):]

            state = new_state

            if global_step > 500:
                idx = np.random.choice(np.arange(1, min(buffer_size,global_step)), 64)
                nn.train(batch_s[idx, :], batch_a[idx], batch_q[idx, :])


        print(step)
        average_reward = total_reward/step

        global_steps.append(global_step)
        performance_curve.append(average_reward)
        steps.append(step)

    PC.append(performance_curve)
    GS.append(global_steps)
    ST.append(steps)

ST = np.float32(np.array(ST))
GS = np.float32(np.array(GS))
PC = np.array(PC)
# plt.plot(noise_OU)
# plt.show()
# plt.close()
#
# plt.plot(global_steps, performance_curve)
# plt.show()

data = {'global_step':GS, 'steps':ST, 'mean_reward':PC}

sio.savemat('../data/cartpole3_batch_WN_' +'.mat', data)


########################## OU noise #################################


env = gym.make('CartPole-v1')

PC = []
GS = []
ST = []

for trial in range(num_trials):

    nn = NN()

    batch_q = np.array([[0]])
    batch_s = np.array([[0,0,0,0]])
    batch_a = np.array([0])

    performance_curve = []
    global_steps = []
    global_step = 0
    explore_step = 0
    steps = []
    noise_OU = 0
    prev_action = 0

    for episode in range(max_episodes):

        state = env.reset()
        done = 0
        step = 0
        total_reward = 0

        while not done and step < 200:

            step += 1
            global_step += 1

            noise_OU = OU_next(noise_OU)

            # tmp = np.random.normal(0, 1)

            if np.abs(noise_OU) < epsilon: # epsilon-greedy

                # action = np.random.choice(2)
                if noise_OU > 0:
                    action = prev_action
                else:
                    action = 1-prev_action
            else:
                action = np.argmax(nn(state))

            new_state, reward, done, _ = env.step(action)
            total_reward += reward

            if not done:
                Q_target = reward + gamma * np.max(nn(new_state))
            else:
                Q_target = reward

            batch_s = np.append(batch_s, np.reshape(state, [1, dim_state]), axis=0)
            batch_a = np.append(batch_a, action)
            batch_q = np.append(batch_q, np.reshape(Q_target, [1, 1]), axis=0)

            if global_step > buffer_size + buffer_size / 5 and global_step % (buffer_size/5) == 0:
                batch_s = batch_s[int(buffer_size / 5):]
                batch_a = batch_a[int(buffer_size / 5):]
                batch_q = batch_q[int(buffer_size / 5):]

            state = new_state

            if global_step > 500:
                idx = np.random.choice(np.arange(1, min(buffer_size,global_step)), 64)
                nn.train(batch_s[idx, :], batch_a[idx], batch_q[idx, :])


        average_reward = total_reward/step
        print(step)

        global_steps.append(global_step)
        performance_curve.append(average_reward)
        steps.append(step)

    PC.append(performance_curve)
    GS.append(global_steps)
    ST.append(steps)

ST = np.float32(np.array(ST))
GS = np.float32(np.array(GS))
PC = np.array(PC)
# plt.plot(noise_OU)
# plt.show()
# plt.close()
#
# plt.plot(global_steps, performance_curve)
# plt.show()

data = {'global_step':GS, 'steps':ST, 'mean_reward':PC}

sio.savemat('../data/cartpole3_batch_OU_' +'.mat', data)

