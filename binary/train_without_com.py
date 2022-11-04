from env_binary import BinaryEnv
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#hyper-parameters
GAMMA = 0.9  # discount factor
LEARNING_RATE = 0.05
EPISODE = 300
STEP = 50
parameter1 = [0.6, 0, 0.4, 0]
parameter2 = [0.8, 0.2, 0, 0]
theta = [0, 0, 0, 0]


class PolicyGradient:
    def __init__(self, beta1, beta2, beta3, beta4):
        self.reward_buffer = []
        self.action_buffer = []
        self.obs_buffer = []
        x1 = random.sample([-1, 1], 1)[0]
        x2 = random.sample([-1, 1], 1)[0]
        self.state = (x1, x2)
        self.obs = [1, 1]
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta3 = beta3
        self.beta4 = beta4

    def get_obs(self, state):
        x = random.random()
        if (x < self.beta1):
            obs = state
        elif (x< self.beta1 + self.beta2):
            obs = (state[0], 0)
        elif (x < self.beta1 + self.beta2 + self.beta3):
            obs = (0, state[1])
        else:
            obs = (0, 0)
        return obs

    def choose_action(self, obs, theta0, theta1):
        rand = random.random()
        x = 1.0/(1.0 + math.exp(- theta0 * obs[0] - theta1 * obs[1]))
        if rand < x:
            action = 1
        else:
            action = 0
        return action

    def store_transition(self, r, a, o):
        self.action_buffer.append(a)
        self.reward_buffer.append(r)
        self.obs_buffer.append(o)

    def learn(self,theta0,theta1):
        self.discounted_reward_buffer = np.zeros_like(self.reward_buffer)
        self.discounted_action_buffer_0 = np.zeros_like(self.reward_buffer)
        self.discounted_action_buffer_1 = np.zeros_like(self.reward_buffer)
        for t in range(len(self.action_buffer)):
            if self.action_buffer[t] == 0:
                self.discounted_action_buffer_0[t] = -self.obs_buffer[t][0]/(
                        1+math.exp(-theta0 * self.obs_buffer[t][0]
                                         - theta1 * self.obs_buffer[t][1]))
                self.discounted_action_buffer_1[t] = -self.obs_buffer[t][1] / (
                            1 + math.exp(-theta0 * self.obs_buffer[t][0]
                                         - theta1 * self.obs_buffer[t][1]))
            elif self.action_buffer[t] == 1:
                self.discounted_action_buffer_0[t] = self.obs_buffer[t][0] * math.exp(-theta0 * self.obs_buffer[t][0]
                                     - theta1 * self.obs_buffer[t][1])/(
                        1 + math.exp(-theta0 * self.obs_buffer[t][0]
                                     - theta1 * self.obs_buffer[t][1]))
                self.discounted_action_buffer_1[t] = self.obs_buffer[t][1] * math.exp(-theta0 * self.obs_buffer[t][0]
                                     - theta1 * self.obs_buffer[t][1]) / (
                        1 + math.exp(-theta0 * self.obs_buffer[t][0]
                                     - theta1 * self.obs_buffer[t][1]))

        for t in range(len(self.reward_buffer)):
            self.discounted_reward_buffer[t] = math.pow(GAMMA, t) * self.reward_buffer[t]
        theta_change_0 = np.sum(self.discounted_reward_buffer * self.discounted_action_buffer_0)
        theta_change_1 = np.sum(self.discounted_reward_buffer * self.discounted_action_buffer_1)
        self.reward_buffer, self.action_buffer, self.obs_buffer = [], [], []
        return LEARNING_RATE * theta_change_0, LEARNING_RATE * theta_change_1, np.sum(self.discounted_reward_buffer)

if __name__ == '__main__':
    list = []
    list2 = []
    list3 = []
    list4 = []
    reward_list = []
    for i in range(5):
        print(i)
        theta1_0 = theta[0]
        theta1_1 = theta[1]
        theta2_0 = theta[2]
        theta2_1 = theta[3]

        env = BinaryEnv()

        agent0 = PolicyGradient(parameter1[0], parameter1[1], parameter1[2], parameter1[3])
        agent1 = PolicyGradient(parameter2[0], parameter2[1], parameter2[2], parameter2[3])

        total_reward = []
        THETA1_0 = []
        THETA1_1 = []
        THETA2_0 = []
        THETA2_1 = []
        episodes = []
        for episode in range(EPISODE):
            state = env.reset()
            episode_nodiscount_reward = 0
            for step in range(STEP):
                obs0 = agent0.get_obs(state)
                obs1 = agent1.get_obs(state)
                action0 = agent0.choose_action(obs0, theta1_0, theta1_1)
                action1 = agent1.choose_action(obs1, theta2_0, theta2_1)
                action = (action0, action1)

                reward, done, state = env.step(action)

                episode_nodiscount_reward += reward
                agent0.store_transition(reward, action0, obs0)
                agent1.store_transition(reward, action1, obs1)

                if done:
                    theta_change1_0, theta_change1_1, episode_reward = agent0.learn(theta1_0, theta1_1)
                    theta_change2_0, theta_change2_1, episode_reward = agent1.learn(theta2_0, theta2_1)

                    theta1_0, theta1_1 = theta1_0 + theta_change1_0, theta1_1 + theta_change1_1
                    theta2_0, theta2_1 = theta2_0 + theta_change2_0, theta2_1 + theta_change2_1

                    total_reward.append(episode_nodiscount_reward)
                    THETA1_0.append(theta1_0)
                    THETA1_1.append(theta1_1)
                    THETA2_0.append(theta2_0)
                    THETA2_1.append(theta2_1)
                    episodes.append(episode)
                    break

        list.append(THETA1_0)
        list2.append(THETA1_1)
        list3.append(THETA2_0)
        list4.append(THETA2_1)
        reward_list.append(total_reward)

    data = pd.DataFrame(data=list)
    data2 = pd.DataFrame(data=list2)
    data3 = pd.DataFrame(data=list3)
    data4 = pd.DataFrame(data=list4)
    re_data = pd.DataFrame(data=reward_list)
    re_data.loc["reward_mean"] = re_data.apply(lambda x: x.mean())
    re_data.loc["reward_std"] = re_data.apply(lambda x: x.std())

    data.loc["THETA1_0_mean"] = data.apply(lambda x: x.mean())
    data.loc["THETA1_0_std"] = data.apply(lambda x: x.std())

    data2.loc["THETA1_1_mean"] = data2.apply(lambda x: x.mean())
    data2.loc["THETA1_1_std"] = data.apply(lambda x: x.std())

    data3.loc["THETA2_0_mean"] = data3.apply(lambda x: x.mean())
    data3.loc["THETA2_0_std"] = data.apply(lambda x: x.std())

    data4.loc["THETA2_1_mean"] = data4.apply(lambda x: x.mean())
    data4.loc["THETA2_1_std"] = data.apply(lambda x: x.std())
    Reward = []
    mean = []
    mean.extend([data.loc["THETA1_0_mean"], data2.loc["THETA1_1_mean"], data3.loc["THETA2_0_mean"], data4.loc["THETA2_1_mean"]])
    mean.extend([data.loc["THETA1_0_std"], data2.loc["THETA1_1_std"], data3.loc["THETA2_0_std"], data4.loc["THETA2_1_std"]])
    Reward.extend([re_data.loc["reward_mean"], re_data.loc["reward_std"]])
    data_mean = pd.DataFrame(data=mean)

    plt.figure()
    plt.plot(episodes, mean[0])
    plt.plot(episodes, mean[1])
    plt.plot(episodes, mean[2], linestyle="--", marker="o", markersize=5, markevery=20)
    plt.plot(episodes, mean[3], linestyle="--", marker="o", markersize=5, markevery=20)
    plt.legend(["THETA1_0", "THETA1_1", "THETA2_0", "THETA2_1"])
    plt.show()

    plt.figure()
    plt.plot(episodes, Reward[0])
    plt.legend(["reward"])
    plt.show()