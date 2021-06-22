import numpy as np
import os

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

# from assets.FeatureExtractor import FeatureExtractor
from assets.FeatureExtractor_2 import FeatureExtractor_2
from assets.ActorNet import ActorNet
from assets.CriticNet import CriticNet
from assets.ReplayMemory import ReplayMemory
from assets.OpticalFlow_s import OpticalFlow_s


class DDPG(object):
    def __init__(
        self,
        obs_shape,
        action_size,
        n_frameStack,
        device,
        GAMMA = 0.96,
        A_LR = 3e-5,
        C_LR = 3e-5,
        TAU = 0.001,
        MEMORY_SIZE = 2000,
        BATCH_SIZE = 128,
        EPSILON = 0.9,
        opticalFlow_ICM = False,
        FLOW_LR = None,
        EXT_R_COE = None,
        INT_R_COE = None,
        flow_update_interval = None,
        TESTING = False,
    ):
        super(DDPG, self).__init__()


        self.obs_shape = obs_shape
        self.action_size = action_size
        

        self.device = device


        self.GAMMA = GAMMA
        self.A_LR = A_LR
        self.C_LR = C_LR
        self.TAU = TAU

        self.EPSILON = EPSILON

        self.BATCH_SIZE = BATCH_SIZE


        self.feature_size = 512
        self.feature_extractor = FeatureExtractor_2(self.obs_shape, self.feature_size)


        self.opticalFlow_ICM = opticalFlow_ICM
        if(self.opticalFlow_ICM):
            self.FLOW_LR = FLOW_LR
            self.EXT_R_COE = EXT_R_COE
            self.INT_R_COE = INT_R_COE

            self.flow_update_interval = flow_update_interval
            self.update_steps = 0

            self.icm = OpticalFlow_s(self.obs_shape, n_frameStack).to(self.device)
            if(not TESTING):
                self.icm_optimizer = optim.Adam(self.icm.parameters(), lr=self.FLOW_LR)


        # actor
        self.actor = ActorNet(self.feature_size, self.feature_extractor, self.action_size).to(self.device)
        if(not TESTING):
            self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.A_LR)

        self.actor_target = ActorNet(self.feature_size, self.feature_extractor, self.action_size).to(self.device)
        self.actor_target.eval()


        # critic
        self.critic = CriticNet(self.feature_size, self.action_size, self.feature_extractor).to(self.device)
        if(not TESTING):
            self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.C_LR)

        self.critic_target = CriticNet(self.feature_size, self.action_size, self.feature_extractor).to(self.device)
        self.critic_target.eval()

        self.copy_target(0)


        self.base_noise = 0.5
        self.NOISE_DECAY = 0.999998


        self.memory = ReplayMemory(MEMORY_SIZE, self.obs_shape, self.action_size)



    def copy_target(self, decay=None):
        if decay is None:
            decay = 1.0 - self.TAU

        # actor
        for param, target_param in zip(self.actor.parameters(),
                                       self.actor_target.parameters()):
            target_param.data.copy_((1 - decay) * param.data +
                                    decay * target_param.data)

        # critic
        for param, target_param in zip(self.critic.parameters(),
                                       self.critic_target.parameters()):
            target_param.data.copy_((1 - decay) * param.data +
                                    decay * target_param.data)


    def update(self):

        obs, action, reward, next_obs, done = self.memory.sample(self.BATCH_SIZE)

        obs = torch.tensor(obs, device=self.device, dtype=torch.float)
        action = torch.tensor(action, device=self.device, dtype=torch.float)
        next_obs = torch.tensor(next_obs, device=self.device, dtype=torch.float)
        done = torch.tensor(done, device=self.device, dtype=torch.bool)
        reward = torch.tensor(reward, device=self.device, dtype=torch.float)
        
        
        if(self.opticalFlow_ICM):
            if(self.update_steps % self.flow_update_interval == 0):
                print("update flow")
                flow_loss, pred_error = self.icm.compute_loss(obs, next_obs, self.device)
                pred_error = pred_error.detach()

                self.icm_optimizer.zero_grad()
                flow_loss.backward()
                for param in self.icm.parameters():
                    param.grad.data.clamp_(-1, 1)
                self.icm_optimizer.step()
            else:
                with torch.no_grad():
                    flow_loss, pred_error = self.icm.compute_loss(obs, next_obs, self.device)

            reward = self.EXT_R_COE*reward + self.INT_R_COE*pred_error



        # critic
        with torch.no_grad():
            target_value = self.critic_target( [next_obs, self.actor_target(next_obs)] )
            target_value[done] = 0
            target_Q = reward + (self.GAMMA * target_value)


        current_Q = self.critic( [obs, action] )

        critic_loss = F.mse_loss(current_Q, target_Q)


        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        for param in self.critic.parameters():
            param.grad.data.clamp_(-1, 1)
        self.critic_optimizer.step()



        # actor
        actor_loss = -self.critic( [obs, self.actor(obs)] ).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        for param in self.actor.parameters():
            param.grad.data.clamp_(-1, 1)
        self.actor_optimizer.step()


        self.copy_target()


        if(self.opticalFlow_ICM):
            self.update_steps += 1
            return critic_loss.item(), actor_loss.item(), flow_loss.item()
        else:
            return critic_loss.item(), actor_loss.item()


    def save(self, path, base):
        torch.save(self.actor.state_dict(), path + "_actor")
        torch.save(self.critic.state_dict(), path + "_critic")
        torch.save(self.feature_extractor.state_dict(), path + "_feature_extractor")


        if(self.opticalFlow_ICM):
            torch.save(self.icm.state_dict(), path + "_icm")
        


    def load(self, path, mem, base):
        print("load from " + path)

        self.actor.load_state_dict(torch.load(path + "_actor"))
        self.critic.load_state_dict(torch.load(path + "_critic"))
        self.feature_extractor.load_state_dict(torch.load(path + "_feature_extractor"))

        self.copy_target(0)


        if(self.opticalFlow_ICM):
            self.icm.load_state_dict(torch.load(path + "_icm"))


    def update_noise(self):
        self.base_noise *= self.NOISE_DECAY


    # # for training (epsilon greedy)
    # def choose_action(self, obs, step_ratio):
    # # def choose_action(self, obs):
    #     epsilon = self.EPSILON + (0.1-self.EPSILON)*step_ratio

    #     # if np.random.uniform() > self.EPSILON:
    #     if np.random.uniform() > epsilon:
    #         with torch.no_grad():
    #             obs = torch.tensor([obs], device=self.device, dtype=torch.float)

    #             # action = self.actor(obs).view(-1).cpu().numpy()
    #             action = self.actor(obs)
    #             action = action.data.max(1, keepdim=True)[1].view(-1).cpu().numpy()[0]

    #             # # add noise for explore
    #             # current_noise = self.base_noise * (0.98**steps)
    #             # action = np.clip(np.random.normal(action, current_noise), -1.0, 1.0)

    #             # action = (action + 1.0) * 0.5
    #             # action = np.clip(action, 0.0, 1.0)
    #     else:
    #         action = np.random.randint(0, self.action_size)
    #     return action


    # for training
    def choose_action(self, obs, steps):

        with torch.no_grad():
            obs = torch.tensor([obs], device=self.device, dtype=torch.float)

            action = self.actor(obs).view(-1).cpu().numpy()

            # add noise for explore
            current_noise = self.base_noise * (0.98**steps)
            action = np.clip(np.random.normal(action, current_noise), -1.0, 1.0)

        return action



    # for testing
    def act(self, obs):
        with torch.no_grad():
            obs = torch.tensor([obs], device=self.device, dtype=torch.float)

            action = self.actor(obs).view(-1).cpu().numpy()

        return action