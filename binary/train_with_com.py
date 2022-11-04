from env_binary import BinaryEnv
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#hyper-parameters
GAMMA = 0.9  # discount factor
LEARNING_RATE1 = 0.05
LEARNING_RATE2 = 0.1
EPISODE = 500
STEP = 50
discount_com = 0.2
parameter1 = [0.6, 0, 0.4, 0]
parameter2 = [0.8, 0.2, 0, 0]
theta = [0, 0, 0, 0]
com_theta = [0, 0, 0, 0]

class PolicyGradient:

    def __init__(self, beta1, beta2, beta3, beta4):
        self.reward_buffer = []
        self.commu_reward_buffer = []
        self.commu_flag_buffer = []
        self.commu_obs_buffer = []
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
        self.commu_count = 0

    def get_obs(self, state):

        x = random.random()
        if (x < self.beta1):
            obs = state
            comm_obs = (1, 1)
        elif (x< self.beta1 + self.beta2):
            obs = (state[0], 0)
            comm_obs = (1, -1)
        elif (x < self.beta1 + self.beta2 + self.beta3):
            obs = (0, state[1])
            comm_obs = (-1, 1)
        else:
            obs = (0, 0)
            comm_obs = (-1, -1)
        return obs, comm_obs

    def choose_communication(self, obs, commu_1, commu_2):

        rand = random.random()
        # print(obs)
        x = 1.0/(1.0 + math.exp(- commu_1 * obs[0] - commu_2 * obs[1]))
        if rand < x:
            commu_flag = 1
        else:
            commu_flag = 0

        return commu_flag

    def obs_communication(self, obs0, obs1):
        obs0 = list(obs0)
        obs1 = list(obs1)
        for t in range(2):

            if obs0[t] == 0:
                obs0[t] = obs1[t]
            if random.random() < 0.25:
                obs0[t] = 0
        return obs0

    def choose_action(self, obs, theta0, theta1):
        rand = random.random()
        x = 1.0/(1.0 + math.exp(- theta0 * obs[0] - theta1 * obs[1]))
        if rand < x:
            action = 1
        else:
            action = 0
        return action

    def store_transition(self, a, r, o, c_r, n_o, c_f):
        self.action_buffer.append(a)
        self.reward_buffer.append(r)
        self.commu_obs_buffer.append(o)
        self.commu_reward_buffer.append(c_r)
        self.obs_buffer.append(n_o)
        self.commu_flag_buffer.append(c_f)

    def learn_communication(self, theta0, theta1):
        self.discounted_reward_buffer = np.zeros_like(self.commu_reward_buffer)
        self.discounted_commu_buffer_0 = np.zeros_like(self.commu_reward_buffer)
        self.discounted_commu_buffer_1 = np.zeros_like(self.commu_reward_buffer)
        for t in range(len(self.commu_flag_buffer)):
            if self.commu_flag_buffer[t] == 0:
                self.discounted_commu_buffer_0[t] = -self.obs_buffer[t][0]/(
                        1+math.exp(-theta0 * self.obs_buffer[t][0]
                                         - theta1 * self.obs_buffer[t][1]))

                self.discounted_commu_buffer_1[t] = -self.obs_buffer[t][1] / (
                            1 + math.exp(-theta0 * self.obs_buffer[t][0]
                                         - theta1 * self.obs_buffer[t][1]))

            elif self.commu_flag_buffer[t] == 1:
                self.discounted_commu_buffer_0[t] = self.obs_buffer[t][0] * math.exp(-theta0 * self.obs_buffer[t][0]
                                     - theta1 * self.obs_buffer[t][1])/(
                        1 + math.exp(-theta0 * self.obs_buffer[t][0]
                                     - theta1 * self.obs_buffer[t][1]))

                self.discounted_commu_buffer_1[t] = self.obs_buffer[t][1] * math.exp(-theta0 * self.obs_buffer[t][0]
                                     - theta1 * self.obs_buffer[t][1]) / (
                        1 + math.exp(-theta0 * self.obs_buffer[t][0]
                                     - theta1 * self.obs_buffer[t][1]))

        for t in range(len(self.commu_reward_buffer)):
            self.discounted_reward_buffer[t] = math.pow(GAMMA, t) * self.commu_reward_buffer[t]

        theta_change_0 = np.sum(self.discounted_reward_buffer * self.discounted_commu_buffer_0)
        theta_change_1 = np.sum(self.discounted_reward_buffer * self.discounted_commu_buffer_1)
        self.commu_reward_buffer, self.commu_flag_buffer, self.obs_buffer = [], [], []

        return LEARNING_RATE2 * theta_change_0, LEARNING_RATE2 * theta_change_1

    def learn(self, theta0, theta1):
        self.discounted_reward_buffer = np.zeros_like(self.reward_buffer)
        self.discounted_action_buffer_0 = np.zeros_like(self.reward_buffer)
        self.discounted_action_buffer_1 = np.zeros_like(self.reward_buffer)

        for t in range(len(self.action_buffer)):

            if self.action_buffer[t] == 0:
                self.discounted_action_buffer_0[t] = -self.commu_obs_buffer[t][0]/(
                        1+math.exp(-theta0 * self.commu_obs_buffer[t][0]
                                         - theta1 * self.commu_obs_buffer[t][1]))

                self.discounted_action_buffer_1[t] = -self.commu_obs_buffer[t][1] / (
                            1 + math.exp(-theta0 * self.commu_obs_buffer[t][0]
                                         - theta1 * self.commu_obs_buffer[t][1]))

            elif self.action_buffer[t] == 1:
                self.discounted_action_buffer_0[t] = self.commu_obs_buffer[t][0] * math.exp(-theta0 * self.commu_obs_buffer[t][0]
                                     - theta1 * self.commu_obs_buffer[t][1])/(
                        1 + math.exp(-theta0 * self.commu_obs_buffer[t][0]
                                     - theta1 * self.commu_obs_buffer[t][1]))

                self.discounted_action_buffer_1[t] = self.commu_obs_buffer[t][1] * math.exp(-theta0 * self.commu_obs_buffer[t][0]
                                     - theta1 * self.commu_obs_buffer[t][1]) / (
                        1 + math.exp(-theta0 * self.commu_obs_buffer[t][0]
                                     - theta1 * self.commu_obs_buffer[t][1]))

        for t in range(len(self.reward_buffer)):
            self.discounted_reward_buffer[t] = math.pow(GAMMA, t) * self.reward_buffer[t]

        theta_change_0 = np.sum(self.discounted_reward_buffer * self.discounted_action_buffer_0)
        theta_change_1 = np.sum(self.discounted_reward_buffer * self.discounted_action_buffer_1)
        self.reward_buffer, self.action_buffer, self.commu_obs_buffer = [], [], []

        return LEARNING_RATE1 * theta_change_0, LEARNING_RATE1 * theta_change_1, np.sum(self.discounted_reward_buffer)

if __name__ == '__main__':

    full_commu = 1
    all_full_commu = 1
    # reward_threshold = 0
    epsidoe_threshold = 0
    list1 = []
    list2 = []
    list3 = []
    list4 = []
    C_list1 = []
    C_list2 = []
    C_list3 = []
    C_list4 = []
    CF_1 = []
    CF_2 = []

    reward_list = []


    for i in range(5):
        print(i)
        theta1_0 = theta[0]
        theta1_1 = theta[1]
        theta2_0 = theta[2]
        theta2_1 = theta[3]
        # --------com setting------- #
        commu1_0 = com_theta[0]
        commu1_1 = com_theta[1]
        commu2_0 = com_theta[2]
        commu2_1 = com_theta[3]

        env = BinaryEnv()
        agent0 = PolicyGradient(parameter1[0], parameter1[1], parameter1[2], parameter1[3])
        agent1 = PolicyGradient(parameter2[0], parameter2[1], parameter2[2], parameter2[3])

        total_reward = []
        THETA1_0 = []
        THETA1_1 = []
        THETA2_0 = []
        THETA2_1 = []

        C_THETA1_0 = []
        C_THETA1_1 = []
        C_THETA2_0 = []
        C_THETA2_1 = []

        A1_commu_frequency = []
        A2_commu_frequency = []

        episodes = []
        learn_commu_flag = 0

        if epsidoe_threshold == 0:
            learn_commu_flag = 1

        for episode in range(EPISODE):
            state = env.reset()
            episode_nodiscount_reward = 0
            agent0.commu_count = 0
            agent1.commu_count = 0
            if all_full_commu == 0:
                if epsidoe_threshold >= episode and epsidoe_threshold != 0:
                    full_commu = 1
                else:
                    full_commu = 0

            for step in range(STEP):
                obs0, commu_obs0 = agent0.get_obs(state)
                obs1, commu_obs1 = agent1.get_obs(state)

                if full_commu == 0:
                    commu_flag0 = agent0.choose_communication(commu_obs0, commu1_0, commu1_1)
                    commu_flag1 = agent1.choose_communication(commu_obs1, commu2_0, commu2_1)
                elif full_commu == 1:
                    commu_flag0 = commu_flag1 = 1
                if learn_commu_flag == 1 or full_commu == 1:
                    if commu_flag0 == 1:
                        obs0 = agent0.obs_communication(obs0, obs1)
                        agent0.commu_count += 1
                    if commu_flag1 == 1:
                        obs1 = agent1.obs_communication(obs1, obs0)
                        agent1.commu_count += 1

                action0 = agent0.choose_action(obs0, theta1_0, theta1_1)
                action1 = agent1.choose_action(obs1, theta2_0, theta2_1)
                action = (action0, action1)

                reward, done, state = env.step(action)

                if learn_commu_flag == 1:
                    commu_reward1 = reward - commu_flag0 * discount_com
                    commu_reward2 = reward - commu_flag1 * discount_com
                else:
                    commu_reward1 = commu_reward2 = 0

                episode_nodiscount_reward += reward

                agent0.store_transition(action0, reward, obs0, commu_reward1,  commu_obs0,  commu_flag0)
                agent1.store_transition(action1, reward, obs1, commu_reward2,  commu_obs1,  commu_flag1)

                if done:
                    theta_change1_0, theta_change1_1, episode_reward = agent0.learn(theta1_0, theta1_1)
                    theta_change2_0, theta_change2_1, episode_reward = agent1.learn(theta2_0, theta2_1)

                    theta1_0, theta1_1 = theta1_0 + theta_change1_0, theta1_1 + theta_change1_1
                    theta2_0, theta2_1 = theta2_0 + theta_change2_0, theta2_1 + theta_change2_1

                    # -------communication-----#
                    if learn_commu_flag == 1:
                        theta_change1_0, theta_change1_1 = agent0.learn_communication(commu1_0, commu1_1)
                        theta_change2_0, theta_change2_1 = agent1.learn_communication(commu2_0, commu2_1)

                        commu1_0, commu1_1 = commu1_0 + theta_change1_0, commu1_1 + theta_change1_1
                        commu2_0, commu2_1 = commu2_0 + theta_change2_0, commu2_1 + theta_change2_1

                    total_reward.append(episode_nodiscount_reward)
                    if episode >= epsidoe_threshold:
                        learn_commu_flag = 1

                    THETA1_0.append(theta1_0)
                    THETA1_1.append(theta1_1)
                    THETA2_0.append(theta2_0)
                    THETA2_1.append(theta2_1)

                    C_THETA1_0.append(commu1_0)
                    C_THETA1_1.append(commu1_1)
                    C_THETA2_0.append(commu2_0)
                    C_THETA2_1.append(commu2_1)

                    episodes.append(episode)
                    break

            A1_commu_frequency.append(agent0.commu_count / STEP)
            A2_commu_frequency.append(agent1.commu_count / STEP)

        CF_1.append(A1_commu_frequency)
        CF_2.append(A2_commu_frequency)

        list1.append(THETA1_0)
        list2.append(THETA1_1)
        list3.append(THETA2_0)
        list4.append(THETA2_1)

        C_list1.append(C_THETA1_0)
        C_list2.append(C_THETA1_1)
        C_list3.append(C_THETA2_0)
        C_list4.append(C_THETA2_1)

        reward_list.append(total_reward)

    data1 = pd.DataFrame(data=list1)
    data2 = pd.DataFrame(data=list2)
    data3 = pd.DataFrame(data=list3)
    data4 = pd.DataFrame(data=list4)
    C_data1 = pd.DataFrame(data=C_list1)
    C_data2 = pd.DataFrame(data=C_list2)
    C_data3 = pd.DataFrame(data=C_list3)
    C_data4 = pd.DataFrame(data=C_list4)

    CF_data1 = pd.DataFrame(data=CF_1)
    CF_data2 = pd.DataFrame(data=CF_2)

    CF_data1.loc["A1_Commu_frequency_mean"] = CF_data1.apply(lambda x: x.mean())
    CF_data1.loc["A1_Commu_frequency_std"] = CF_data1.apply(lambda x: x.std())
    CF_data2.loc["A2_Commu_frequency_mean"] = CF_data2.apply(lambda x: x.mean())
    CF_data2.loc["A2_Commu_frequency_std"] = CF_data2.apply(lambda x: x.std())

    re_data = pd.DataFrame(data=reward_list)
    re_data.loc["reward_mean"] = re_data.apply(lambda x: x.mean())
    re_data.loc["reward_std"] = re_data.apply(lambda x: x.std())

    data1.loc["THETA1_0_mean"] = data1.apply(lambda x: x.mean())
    data1.loc["THETA1_0_std"] = data1.apply(lambda x: x.std())

    data2.loc["THETA1_1_mean"] = data2.apply(lambda x: x.mean())
    data2.loc["THETA1_1_std"] = data2.apply(lambda x: x.std())

    data3.loc["THETA2_0_mean"] = data3.apply(lambda x: x.mean())
    data3.loc["THETA2_0_std"] = data3.apply(lambda x: x.std())

    data4.loc["THETA2_1_mean"] = data4.apply(lambda x: x.mean())
    data4.loc["THETA2_1_std"] = data4.apply(lambda x: x.std())

    C_data1.loc["C_THETA1_0_mean"] = C_data1.apply(lambda x: x.mean())
    C_data2.loc["C_THETA1_1_mean"] = C_data2.apply(lambda x: x.mean())
    C_data3.loc["C_THETA2_0_mean"] = C_data3.apply(lambda x: x.mean())
    C_data4.loc["C_THETA2_1_mean"] = C_data4.apply(lambda x: x.mean())

    Reward = []
    mean = []
    mean.extend([data1.loc["THETA1_0_mean"], data2.loc["THETA1_1_mean"],
                 data3.loc["THETA2_0_mean"], data4.loc["THETA2_1_mean"]])

    mean.extend([data1.loc["THETA1_0_std"], data2.loc["THETA1_1_std"],
                 data3.loc["THETA2_0_std"], data4.loc["THETA2_1_std"]])

    mean.extend([C_data1.loc["C_THETA1_0_mean"], C_data2.loc["C_THETA1_1_mean"],
                 C_data3.loc["C_THETA2_0_mean"], C_data4.loc["C_THETA2_1_mean"]])

    mean.extend([CF_data1.loc["A1_Commu_frequency_mean"], CF_data2.loc["A2_Commu_frequency_mean"],
                 CF_data1.loc["A1_Commu_frequency_std"], CF_data2.loc["A2_Commu_frequency_std"]])

    Reward.extend([re_data.loc["reward_mean"], re_data.loc["reward_std"]])

    data_mean = pd.DataFrame(data=mean)

    plt.figure()
    plt.plot(episodes, mean[0])
    plt.plot(episodes, mean[1])
    plt.plot(episodes, mean[2], linestyle="--", marker="o", markersize=5, markevery=20)
    plt.plot(episodes, mean[3], linestyle="--", marker="o", markersize=5, markevery=20)
    plt.legend([r'$\theta_{11}$', r'$\theta_{12}$', r'$\theta_{21}$', r'$\theta_{22}$'])
    plt.show()

    plt.figure()
    plt.plot(episodes, Reward[0])
    plt.legend(["reward"])
    plt.show()

    plt.figure()
    plt.plot(episodes, mean[8])
    plt.plot(episodes, mean[9])
    plt.plot(episodes, mean[10], linestyle="--",marker="o", markersize=5, markevery=20)
    plt.plot(episodes, mean[11], linestyle="--",marker="o", markersize=5, markevery=20)
    plt.legend([r'$\omega_{11}$', r'$\omega_{12}$', r'$\omega_{21}$', r'$\omega_{22}$'])
    plt.show()

    plt.figure()
    plt.plot(episodes, mean[12])
    plt.plot(episodes, mean[13], linestyle="--")
    plt.legend(["A1_Commu_Frequence", "A2_Commu_Frequence"])
    plt.show()