import torch
import torch.nn as nn
import torch.nn.functional as F


class CriticNet(nn.Module):
    # def __init__(self, body_obs_size, vel_obs_size, action_size):
    def __init__(self, feature_size, action_size, feature_extractor, hidden1=512, hidden2=512):
        super(CriticNet, self).__init__()

        # self.body_obs_size = body_obs_size
        # self.vel_obs_size = vel_obs_size
        # self.action_size = action_size


        # self.fc0 = nn.Linear(self.body_obs_size, 800)
        # self.fc1 = nn.Linear(800, 400)


        # self.vel_fc0 = nn.Linear(self.vel_obs_size, 200)
        # self.vel_fc1 = nn.Linear(200, 400)


        # self.act_fc0 = nn.Linear(self.action_size, 400)


        # self.fc2 = nn.Linear(400+400+400, 400)
        # self.fc3 = nn.Linear(400, 1)

        self.feature_size = feature_size
        self.action_size = action_size

        self.feature_extractor = feature_extractor

        self.fc1 = nn.Linear(feature_size, hidden1)

        self.fc2 = nn.Linear(hidden1 + self.action_size, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)


    def forward(self, x):
        # body_obs = x[:,:-(self.vel_obs_size+self.action_size)]
        # vel_obs = x[:,-(self.vel_obs_size+self.action_size):-self.action_size]
        # action = x[:,-self.action_size:]

        # body_obs = F.selu(self.fc0(body_obs))
        # body_obs = F.selu(self.fc1(body_obs))

        # vel_obs = F.selu(self.vel_fc0(vel_obs))
        # vel_obs = F.selu(self.vel_fc1(vel_obs))

        # action = F.selu(self.act_fc0(action))

        # # cat_feature = torch.cat((body_obs, action, vel_obs), 1)
        # cat_feature = torch.cat((body_obs, vel_obs, action), 1)

        # cat_feature = F.selu(self.fc2(cat_feature))
        # cat_feature = F.selu(self.fc3(cat_feature))

        # return cat_feature.squeeze(1)


        obs, action = x


        obs = self.feature_extractor(obs)
        obs = F.relu(self.fc1(obs))

        x = torch.cat((obs, action), 1)

        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x.squeeze(1)


    # def value(self, obs, action):
    #     obs = self.feature_extractor(obs)
    #     obs = F.relu(self.fc1(obs))

    #     x = torch.cat((obs, action), 1)
    #     return self.forward(x)