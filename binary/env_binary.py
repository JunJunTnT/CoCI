import random

Step = 50
class BinaryEnv():

    def __init__(self):
        self.statelist = [(-1, -1), (1, -1), (-1, 1), (1, 1)]
        self.action_space = ['-1', '1']
        self.observation_space = ['-1', '1']
        self.count = 0
        x1 = random.sample([-1, 1], 1)[0]
        x2 = random.sample([-1, 1], 1)[0]
        self.state = (x1, x2)

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

    def change_state(self, state, flag):
        x = random.randint(0, 3)
        state = self.statelist[x]
        return state

    def step(self, action):
        action1 = action[0]
        action2 = action[1]

        if (self.state == self.statelist[0] and action1 == action2 == 0) or \
                (self.state == self.statelist[1] and action1 == 1 and action2 == 0) or \
                (self.state == self.statelist[2] and action1 == 0 and action2 == 1) or \
                (self.state == self.statelist[3] and (action1 == action2 == 1)):
            reward = 1.0
            flag = 1
        else:
            reward = 0.0
            flag = 0

        self.state = self.change_state(self.state, flag)
        done = 0
        self.count += 1
        if self.count == Step:
            done = 1

        return reward, done, self.state

    def reset(self):
        x1 = random.sample([-1, 1], 1)[0]
        x2 = random.sample([-1, 1], 1)[0]
        self.state = (x1, x2)
        self.count = 0
        return self.state

