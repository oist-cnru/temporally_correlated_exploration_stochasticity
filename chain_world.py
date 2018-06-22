import numpy as np
import scipy.io as sio

alpha = 0.2
epsilon = 0.1
gamma = 0.99

num_trials = 100
num_episodes = 20000

def OU_next(y_prev, mu=0, sig=1, th=0.15):
    y_next = y_prev + th * (mu - y_prev) + sig * np.sqrt(2) * np.sqrt(th) * np.random.normal()
    return y_next


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
ST = []

for trial in range(num_trials):

    Q_table = np.zeros([7,2])

    performance_curve = []
    global_steps = []
    global_step = 0
    steps = []

    for episode in range(num_episodes):

        state = env.reset()
        done = 0
        Q_table_old = Q_table
        step = 0
        total_reward = 0


        while not done:

            step += 1
            global_step += 1

            tmp = np.random.normal(0.0, 1.0)

            if np.abs(tmp) < epsilon: # epsilon-greedy
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
        steps.append(step)

    PC.append(performance_curve)
    GS.append(global_steps)
    ST.append(steps)

ST = np.float32(np.array(ST))
GS = np.array(GS)
PC = np.array(PC)
# plt.plot(noise_OU)
# plt.show()
# plt.close()
#
# plt.plot(global_steps, performance_curve)
# plt.show()

data = {'global_step':GS, 'steps':ST, 'mean_reward':PC}

sio.savemat('../data/chain_world_WN.mat', data)





########################## OU process as noise #################################

env = Chain_World()

PC = []
GS = []
ST = []

for trial in range(num_trials):


    Q_table = np.zeros([7,2])

    performance_curve = []
    global_steps = []
    global_step = 0
    steps = []
    noise_OU = 0

    for episode in range(num_episodes):

        state = env.reset()
        done = 0
        Q_table_old = Q_table
        step = 0
        total_reward = 0


        while not done:

            step += 1
            global_step += 1

            noise_OU = OU_next(noise_OU)

            if np.abs(noise_OU) < epsilon: # epsilon-greedy
                # action = np.random.choice(2)
                if noise_OU > 0:
                    action = 1
                else:
                    action = 0
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
        steps.append(step)

    PC.append(performance_curve)
    GS.append(global_steps)
    ST.append(steps)

ST = np.float32(np.array(ST))
GS = np.array(GS)
PC = np.array(PC)
# plt.plot(noise_OU)
# plt.show()
# plt.close()
#
# plt.plot(global_steps, performance_curve)
# plt.show()

data = {'global_step':GS, 'steps':ST, 'mean_reward':PC}

sio.savemat('../data/chain_world_OU.mat', data)
