import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorNet(nn.Module):
    # def __init__(self, body_obs_size, vel_obs_size, action_size):
    def __init__(self, feature_size, feature_extractor, output_size, hidden1=512, hidden2=512):
        super(ActorNet, self).__init__()

        # self.body_obs_size = body_obs_size
        # self.vel_obs_size = vel_obs_size


        # self.fc0 = nn.Linear(self.body_obs_size, 800)
        # self.fc1 = nn.Linear(800, 400)


        # self.vel_fc0 = nn.Linear(self.vel_obs_size, 200)
        # self.vel_fc1 = nn.Linear(200, 400)


        # self.fc2 = nn.Linear(800, 200)
        # self.fc3 = nn.Linear(200, action_size)

        self.feature_size = feature_size
        self.output_size = output_size

        self.feature_extractor = feature_extractor

        self.fc1 = nn.Linear(self.feature_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, self.output_size)



    def forward(self, x):
        # body_obs = x[:,:-self.vel_obs_size]
        # vel_obs = x[:,-self.vel_obs_size:]


        # body_obs = torch.tanh(self.fc0(body_obs))
        # body_obs = torch.tanh(self.fc1(body_obs))

        # vel_obs = torch.tanh(self.vel_fc0(vel_obs))
        # vel_obs = torch.tanh(self.vel_fc1(vel_obs))

        # cat_feature = torch.cat((body_obs, vel_obs), 1)

        # cat_feature = torch.tanh(self.fc2(cat_feature))
        # cat_feature = torch.tanh(self.fc3(cat_feature))

        # return cat_feature

        x = self.feature_extractor(x)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # x = torch.tanh(self.fc3(x))
        x = F.softmax(self.fc3(x), dim=1)

        return x