import numpy as np
import random

class DroneAgent:
    # Brain Shape: 10 state dimensions + 5 action outputs
    q_table = np.zeros((3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 5)) 

    def __init__(self, env_size):
        self.epsilon = 1.0
        self.lr = 0.05
        self.gamma = 0.95
        self.actions = [0, 1, 2, 3, 4]  # 0=Up, 1=Right, 2=Down, 3=Left, 4=Hover
        
    def _get_idx(self, state):
        dx, dy, wu, wr, wd, wl, eu, er, ed, el = state
        return (dx + 1, dy + 1, wu, wr, wd, wl, eu, er, ed, el)

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        idx = self._get_idx(state)
        q_vals = self.q_table[idx]
        max_val = np.max(q_vals)
        best_actions = np.where(q_vals == max_val)[0]
        return int(random.choice(best_actions))

    def learn(self, state, action, reward, next_state):
        idx = self._get_idx(state)
        next_idx = self._get_idx(next_state)
        
        predict = self.q_table[idx][action]
        target = reward + self.gamma * np.max(self.q_table[next_idx])
        
        self.q_table[idx][action] += self.lr * (target - predict)
        
