import sys
import pickle
import random
import math
import time

import numpy as np
from collections import namedtuple
from itertools import count
from collections import deque
from tqdm import tqdm

import gym
import gym.spaces
from gym import spaces
import marlo

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd

from assets.EnvWrapper import FrameStack, WarpFrame, StackTranspose
from assets.NoisyNet import NoisyLinear

class DQN(nn.Module):
    def __init__(self, in_channels=3, num_actions=8):
        
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.fc_h_v = NoisyLinear(3136, 512)
        self.fc_z_v = NoisyLinear(512, 1)

        self.fc_h_a = NoisyLinear(3136, 512)
        self.fc_z_a = NoisyLinear(512, num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1,3136)
        
        v = F.relu(self.fc_h_v(x))
        v = self.fc_z_v(v)

        a = F.relu(self.fc_h_a(x))
        a = self.fc_z_a(a)

        q = v + a - a.mean()
        return q

    def reset_noise(self):
        self.fc_h_v.reset_noise()
        self.fc_z_v.reset_noise()
        self.fc_h_a.reset_noise()
        self.fc_z_a.reset_noise()
PATH = './cliffWalking/DQN_CliffWalking.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Agent(object):
    """Agent that acts randomly."""
    def __init__(self):
        self.learning_DQN = DQN(in_channels=3, num_actions=8).to(device)
        self.device = device
        self.learning_DQN.load_state_dict(torch.load(PATH))
    
    def select_action(self,state):
        state_tensor = VectortoTensor(state).cuda()
        prediction = self.learning_DQN(state_tensor)
        return prediction.argmax().item()

def VectortoTensor(state):
    return torch.tensor([state], dtype=torch.float32).cuda()
def test():
    #init
    client_pool = [('127.0.0.1', 10000)]

    join_tokens = marlo.make('MarLo-CliffWalking-v0',
                        params={
                        "client_pool": client_pool,
                                                    
                        # "retry_sleep" : 4,
                        # "step_sleep" : 0.004,
                        "prioritise_offscreen_rendering": False
                        })

    assert len(join_tokens) == 1
    join_token = join_tokens[0]

    env = marlo.init(join_token)
    env = WarpFrame(env)
    env = FrameStack(env, 1)
    env = StackTranspose(env)
    action_space = env.action_space.n

    done = False
    total_reward = 0
    agent = Agent()
    #Training
    epoch = 10
    steps_done = 0

    for i in tqdm(range(epoch)):
        episode_reward_sum = 0
        state = env.reset() #reset state frame
        while True:

            a = agent.select_action(state) #epsilon greedy
            a = 1
            next_state, r, done, info = env.step(a) #step to next state
            #Data To Tensor          
            episode_reward_sum += r
            steps_done += 1
            
            #state <- next_state 
            state = next_state
            if done:
                print('episode%s---reward_sum: %s' % (i, round(episode_reward_sum, 2)))
                print("steps_done: ",steps_done)
                break

if __name__ == "__main__":
    test()
