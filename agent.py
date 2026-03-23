import numpy as np
import random

class DroneAgent:
    def __init__(self, env_size):
        self.size = env_size
        self.q_table = np.zeros((env_size, env_size, 4))

        self.actions = [0, 1, 2, 3]   # numeric actions

        self.epsilon = 0.9
        self.alpha = 0.1
        self.gamma = 0.9

    def choose_action(self, state):
        x, y = state

        if random.random() < self.epsilon:
            return random.choice(self.actions)
        else:
            return np.argmax(self.q_table[x, y])

    def learn(self, state, action, reward, next_state):
        x, y = state
        nx, ny = next_state

        q_predict = self.q_table[x, y, action]
        q_target = reward + self.gamma * np.max(self.q_table[nx, ny])

        self.q_table[x, y, action] += self.alpha * (q_target - q_predict)

        # epsilon decay
        self.epsilon = max(0.05, self.epsilon * 0.995)
