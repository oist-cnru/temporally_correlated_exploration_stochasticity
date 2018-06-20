import tensorflow as tf
from tensorflow.contrib import layers
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.io as sio
import time
import argparse
from copy import deepcopy
from datetime import datetime

alpha = 0.2
epsilon = 0.05
gamma = 0.99

def OUprocess(n, mu=0, sig=1, th=0.3):
    y = np.zeros(n+1)
    for i in range(n):
        y[i + 1] = y[i] + th * (mu - y[i]) + sig * np.sqrt(2) * np.sqrt(th) * np.random.normal()
    return y[1:n]


class Chain_World():

    def __init__(self):
        self.reward_list = np.array([1, 0, 0, 0, -1, 0, 5])
        self.init_state = 3

    def reset(self):
        self.state = self.init_state
        return self.state

    def step(self, a):
        if a == 1:
            self.state += 1
        elif a == 0:
            self.state -= 1

        reward = self.reward_list[self.state]

        if self.state == 0 or self.state == 6:
            done = 1
        else:
            done = 0

        return self.state, reward, done





########################## white noise #################################



env = Chain_World()

PC = []
GS = []

for trial in range(100):

    Q_table = np.zeros([7,2])

    performance_curve = []
    global_steps = []
    global_step = 0

    for episode in range(20000):

        state = env.reset()
        done = 0
        Q_table_old = Q_table
        step = 0
        total_reward = 0


        while not done:

            step += 1
            global_step += 1

            tmp = np.random.rand()

            if tmp < epsilon: # epsilon-greedy
                action = np.random.choice(2)

            else:
                action = np.argmax(Q_table[state])

            new_state, reward, done = env.step(action)
            total_reward += reward

            Q_table[state][action] += alpha * (reward + gamma * max(Q_table[new_state]) - Q_table[state][action] )

            state = new_state

        average_reward = total_reward/step
        # print(step)

        global_steps.append(global_step)
        performance_curve.append(average_reward)
    PC.append(performance_curve)

GS = np.array(GS)
PC = np.array(PC)
# plt.plot(noise_OU)
# plt.show()
# plt.close()
#
# plt.plot(global_steps, performance_curve)
# plt.show()

data = {'step':GS, 'mean_reward':PC}

sio.savemat('../data/chain_world_WN.mat', data)





########################## OU process as noise #################################

env = Chain_World()

PC = []
GS = []

for trial in range(100):

    Q_table = np.zeros([7,2])

    performance_curve = []
    global_steps = []
    global_step = 0
    explore_step = 0

    noise_OU = OUprocess(100000)

    for episode in range(20000):

        state = env.reset()
        done = 0
        Q_table_old = Q_table
        step = 0
        total_reward = 0


        while not done:

            step += 1
            global_step += 1

            tmp = np.random.rand()

            if tmp < epsilon: # epsilon-greedy
                # action = np.random.choice(2)
                explore_step += 1
                if noise_OU[explore_step] > 0:
                    action = 0
                else:
                    action = 1
            else:
                action = np.argmax(Q_table[state])

            new_state, reward, done = env.step(action)
            total_reward += reward

            Q_table[state][action] += alpha * (reward + gamma * max(Q_table[new_state]) - Q_table[state][action] )

            state = new_state

        average_reward = total_reward/step
        # print(step)

        global_steps.append(global_step)
        performance_curve.append(average_reward)
    PC.append(performance_curve)

GS = np.array(GS)
PC = np.array(PC)
# plt.plot(noise_OU)
# plt.show()
# plt.close()
#
# plt.plot(global_steps, performance_curve)
# plt.show()

data = {'step':GS, 'mean_reward':PC}

sio.savemat('../data/chain_world_OU.mat', data)
