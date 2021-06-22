import numpy as np


class ReplayMemory(object):
    def __init__(self, max_size, obs_dim, act_dim):
        self.max_size = int(max_size)
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        # self.obs = np.zeros((max_size, obs_dim[0], obs_dim[1], obs_dim[2]), dtype='float32')
        self.obs = np.zeros((max_size, *obs_dim), dtype='float32')
        self.action = np.zeros((max_size, act_dim), dtype='float32')
        self.reward = np.zeros((max_size, ), dtype='float32')
        self.done = np.zeros((max_size, ), dtype='bool')
        # self.next_obs = np.zeros((max_size, obs_dim[0], obs_dim[1], obs_dim[2]), dtype='float32')
        self.next_obs = np.zeros((max_size, *obs_dim), dtype='float32')

        self.curr_size = 0
        self.curr_pos = 0

    def sample(self, batch_size):
        # batch_idx = np.random.randint(self.curr_size - 300 - 1, size=batch_size)
        batch_idx = np.random.randint(self.curr_size - 1, size=batch_size)

        obs = self.obs[batch_idx]
        reward = self.reward[batch_idx]
        action = self.action[batch_idx]
        next_obs = self.next_obs[batch_idx]
        done = self.done[batch_idx]
        return obs, action, reward, next_obs, done

    def append(self, obs, act, reward, next_obs, done):
        if self.curr_size < self.max_size:
            self.curr_size += 1

        self.obs[self.curr_pos] = obs
        self.action[self.curr_pos] = act
        self.reward[self.curr_pos] = reward
        self.next_obs[self.curr_pos] = next_obs
        self.done[self.curr_pos] = done

        self.curr_pos = (self.curr_pos + 1) % self.max_size

    def size(self):
        return self.curr_size