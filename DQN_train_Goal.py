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
from assets.prioritized_replay_memory_multistep import SumTree,PrioritiedReplayMemory
from assets.NoisyNet import NoisyLinear

exp_rate = 0.1
gamma = 0.999
lr = 0.000125
BATCH_SIZE = 256
steps_done = 0

PATH = ''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)
#replay memory
from collections import namedtuple
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward','done'))

class DQN(nn.Module):
    def __init__(self, in_channels=3, num_actions=7):
        
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
    
class Agent(object):
    """Agent that acts randomly."""
    def __init__(self,exp_rate,gamma,lr,num_actions):
        self.action_space = gym.spaces.Discrete(num_actions)
        self.exp_rate = exp_rate
        self.gamma = gamma
        self.learning_DQN = DQN(in_channels=3, num_actions = num_actions).to(device)
        self.target_DQN = DQN(in_channels=3, num_actions = num_actions).to(device)
        self.optimizer = torch.optim.Adam(self.learning_DQN.parameters(), lr=lr)
        self.prioritiedReplayMemory = PrioritiedReplayMemory(10000,20)
    def load(self):
        self.learning_DQN.load_state_dict(torch.load('./data/Goal/no_ICM/DQN_goal84000.pth'))
        self.learning_DQN.load_state_dict(torch.load('./data/Goal/no_ICM/DQN_goal84000.pth'))
    def random_act(self,state):
        return self.action_space.sample()
    def get_Q_max(self,state):
        state_tensor = VectortoTensor(state).cuda()
        prediction = self.target_DQN(state_tensor)
        return prediction.max().item()
    def get_Q_target(self,state,action):
        state_tensor = VectortoTensor(state).cuda()
        prediction = self.target_DQN(state_tensor)
        return prediction[0][action].item()
    def get_Q_learning(self,state,action):
        state_tensor = VectortoTensor(state).cuda()
        prediction = self.learning_DQN(state_tensor)
        return prediction[0][action].item()
    def select_action(self,state):
        state_tensor = VectortoTensor(state).cuda()
        prediction = self.learning_DQN(state_tensor)
        return prediction.argmax().item()
    def learning_action(self,state): #choose from DQN
        state_tensor = VectortoTensor(state).cuda()
        prediction = self.learning_DQN(state_tensor)
        return prediction.argmax().item()
    def target_action(self,state):
        state_tensor = VectortoTensor(state).cuda()
        prediction = self.target_DQN(state_tensor)
        return prediction.argmax().item()

def VectortoTensor(state):
    return torch.tensor([state], dtype=torch.float32).cuda()

def optimize_model(agent):
    #return if data isn't enough
    if len(agent.prioritiedReplayMemory) < 10000:
        return
    transitions, idxs, is_weights = agent.prioritiedReplayMemory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])    
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    
    reward_batch = torch.cat(batch.reward)
    action_batch = action_batch.view(BATCH_SIZE,1)
    
    state_action_values = agent.learning_DQN(state_batch).gather(1, action_batch) #take state_action_values
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    #take next_states from learning DQN
    next_action_batch = agent.learning_DQN(non_final_next_states).max(1)[1].view(-1,1)
    next_state_values[non_final_mask] = agent.target_DQN(non_final_next_states).gather(1, next_action_batch).view(-1)
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * agent.gamma) + reward_batch
    errors = torch.abs(state_action_values.squeeze(1) - expected_state_action_values).cpu().data.numpy()
    #update sum tree
    for i in range(BATCH_SIZE):
        idx = idxs[i]
        agent.prioritiedReplayMemory.update(idx, errors[i]+ 1e-5)
    # Compute Huber loss
    loss  = (state_action_values.squeeze(1) - expected_state_action_values.detach()).pow(2) * torch.tensor(is_weights).cuda()
    loss  = loss.mean()

    agent.optimizer.zero_grad()
    loss.backward()
    for param in agent.learning_DQN.parameters():
        param.grad.data.clamp_(-1, 1)

    agent.optimizer.step()
    agent.learning_DQN.reset_noise()
    agent.target_DQN.reset_noise()

def train():
    #init
    client_pool = [('127.0.0.1', 10000)]

    join_tokens = marlo.make('MarLo-FindTheGoal-v0',
                        params={
                        "client_pool": client_pool,
                                                    
                        # "retry_sleep" : 4,
                        # "step_sleep" : 0.004,
                        # "prioritise_offscreen_rendering": False
                        })
    # As this is a single agent scenario,
    # there will just be a single token
    assert len(join_tokens) == 1
    join_token = join_tokens[0]

    env = marlo.init(join_token)
    env = WarpFrame(env)
    env = FrameStack(env, 1)
    env = StackTranspose(env)
    action_space = env.action_space.n
    obs_shape = env.observation_space.shape
    print(obs_shape)
    done = False
    total_reward = 0
    agent = Agent(exp_rate = 0.0,gamma = 0.99,lr = 0.000125,num_actions = action_space)
    
    #Training
    epoch = 1000000001
    steps_done = 0

    for i in tqdm(range(epoch)):
        episode_reward_sum = 0
        state = env.reset() #reset state frame
        while True:

            a = agent.select_action(state) #epsilon greedy
            next_state, r, done, info = env.step(a) #step to next state
            #calculate TD error
            pred = agent.get_Q_learning(state,a)
            next_action = agent.learning_action(next_state)
            target = agent.get_Q_target(next_state,next_action)
            target_value = r + (1 - done)*gamma*target
            error = abs(target_value - pred)
            #Data To Tensor
            action = torch.tensor([a], device=device)
            state_tensor,next_state_tensor = VectortoTensor(state),VectortoTensor(next_state)

            reward = torch.tensor([r], device=device)
            agent.prioritiedReplayMemory.push(error,state_tensor, action, next_state_tensor, reward, done)
            
            episode_reward_sum += (r)
            steps_done += 1
            
            optimize_model(agent)
            #state <- next_state 
            state = next_state
            if(steps_done % 10000 == 0 and steps_done > 1):
                agent.target_DQN.load_state_dict(agent.learning_DQN.state_dict())
            if done:
                print('episode%s---reward_sum: %s' % (i, round(episode_reward_sum, 2)))
                print("steps_done: ",steps_done)
                break
            if(steps_done % 2000 == 0 and i > 1): #save weight
                torch.save(agent.learning_DQN.state_dict(), './data/Goal/no_ICM/DQN_Goal' + str(steps_done) + '.pth')
def main():
    train()

if __name__ == "__main__":
    main()

    