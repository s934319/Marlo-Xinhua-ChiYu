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
from assets.Utils import random_agent_obs_mean_std
from assets.OpticalFlow_s import OpticalFlow_s

exp_rate = 0.1
gamma = 0.999
lr = 0.000125
BATCH_SIZE = 64
steps_done = 0

FLOW_LR = 1e-6
EXT_R_COE = 0.5
INT_R_COE = 0.5
flow_update_interval = 4

PATH = ''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)
#replay memory
from collections import namedtuple
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward','done'))

def VectortoTensor(state):
    return torch.tensor([state], dtype=torch.float32).cuda()

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
    def __init__(self,gamma,lr,FLOW_LR,EXT_R_COE,INT_R_COE,obs_shape,obs_mean,obs_std,flow_update_interval,n_frameStack = 1):
        self.action_space = gym.spaces.Discrete(5)
        self.gamma = gamma
        self.learning_DQN = DQN(in_channels=3, num_actions=5).to(device)
        self.target_DQN = DQN(in_channels=3, num_actions=5).to(device)
        self.optimizer = torch.optim.Adam(self.learning_DQN.parameters(), lr=lr)
        self.prioritiedReplayMemory = PrioritiedReplayMemory(10000)
        self.device = device

        #ICM intrinsic reward
        self.FLOW_LR = FLOW_LR
        self.EXT_R_COE = EXT_R_COE
        self.INT_R_COE = INT_R_COE

        self.flow_update_interval = 4
        self.update_steps = 0
        self.n_frameStack = n_frameStack

        self.obs_shape = obs_shape
        self.obs_mean = obs_mean
        self.obs_std = obs_std

        self.icm = OpticalFlow_s(self.obs_shape, self.obs_mean, self.obs_std, self.n_frameStack).to(self.device)
        self.icm_optimizer = optim.Adam(self.icm.parameters(), lr=self.FLOW_LR)
    
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

    def optimize_model(self):
        #return if data isn't enough
        if len(self.prioritiedReplayMemory) < BATCH_SIZE + 1:
            return
        transitions, idxs, is_weights = self.prioritiedReplayMemory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
        
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat(batch.next_state)
        action_batch = action_batch.view(BATCH_SIZE,1)
        #ICM reward
        if(self.update_steps % self.flow_update_interval == 0):
            flow_loss, pred_error = self.icm.compute_loss(state_batch, next_state_batch, self.device)
            pred_error = pred_error.detach()

            self.icm_optimizer.zero_grad()
            flow_loss.backward()
            for param in self.icm.parameters():
                param.grad.data.clamp_(-1, 1)
            self.icm_optimizer.step()
        else:
            with torch.no_grad():
                flow_loss, pred_error = self.icm.compute_loss(state_batch, next_state_batch, self.device)

        reward_batch = self.EXT_R_COE*reward_batch + self.INT_R_COE*pred_error*0.0025
        
        state_action_values = self.learning_DQN(state_batch).gather(1, action_batch) #take state_action_values
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        #take next_states from learning DQN
        next_action_batch = self.learning_DQN(non_final_next_states).max(1)[1].view(-1,1)
        next_state_values[non_final_mask] = self.target_DQN(non_final_next_states).gather(1, next_action_batch).view(-1)
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        errors = torch.abs(state_action_values.squeeze(1) - expected_state_action_values).cpu().data.numpy()
        
        #update sum tree
        for i in range(BATCH_SIZE):
            idx = idxs[i]
            self.prioritiedReplayMemory.update(idx, errors[i]+ 1e-5)
        # Compute Huber loss
        loss  = (state_action_values.squeeze(1) - expected_state_action_values.detach()).pow(2) * torch.tensor(is_weights).cuda()
        loss  = loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.learning_DQN.parameters():
            param.grad.data.clamp_(-1, 1)

        self.optimizer.step()
        self.learning_DQN.reset_noise()
        self.target_DQN.reset_noise()
        self.update_steps += 1

def train():
    #init
    client_pool = [('127.0.0.1', 10000)]

    join_tokens = marlo.make('MarLo-MazeRunner-v0',
                        params={
                        "client_pool": client_pool,
                        "seed":654684,
                        "maze_height":2                         
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

    obs_shape = env.observation_space.shape

    load = False
    if(load == False):
        obs_mean, obs_std = random_agent_obs_mean_std(env,10000)
        torch.save(obs_mean, "./mazeRunner/" + "maze_obs_mean")
        torch.save(obs_std, "./mazeRunner/" + "maze_obs_std")
    else:
        obs_mean = torch.load("./mazeRunner/" + "maze_obs_mean")
        obs_std = torch.load("./mazeRunner/" + "maze_obs_std")

    done = False
    total_reward = 0

    agent = Agent(gamma = 0.99,lr = 0.000125,FLOW_LR = FLOW_LR,EXT_R_COE = EXT_R_COE,INT_R_COE = INT_R_COE,\
                  flow_update_interval = flow_update_interval,obs_shape = obs_shape,obs_mean = obs_mean,obs_std = obs_std)
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
            agent.prioritiedReplayMemory.push(error,state_tensor, action, next_state_tensor, reward,done) 
            
            episode_reward_sum += r
            steps_done += 1
            
            agent.optimize_model()
            #state <- next_state 
            state = next_state
            if(steps_done % 10000 == 0 and steps_done > 1):
                agent.target_DQN.load_state_dict(agent.learning_DQN.state_dict())
            if done:
                print('episode%s---reward_sum: %s' % (i, round(episode_reward_sum, 2)))
                print("steps_done: ",steps_done)
                break
            if(steps_done % 1000 == 0 and i > 1): #save weight
                torch.save(agent.learning_DQN.state_dict(), './mazeRunner/' + str(steps_done) + '.pth')
                #torch.save(agent.obs_mean.cpu().numpy(), "./data/" + str(steps_done) + "_obs_mean")
                #torch.save(agent.obs_std.cpu().numpy(), "./data/",str(steps_done) + "_obs_std")
                torch.save(agent.icm.state_dict(), './mazeRunner/' + str(steps_done) + "_icm")
                
def main():
    train()

if __name__ == "__main__":
    main()

    