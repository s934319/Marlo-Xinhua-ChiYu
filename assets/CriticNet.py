import torch
import torch.nn as nn
import torch.nn.functional as F


class CriticNet(nn.Module):
    def __init__(self, feature_size, action_size, feature_extractor, hidden1=512, hidden2=512):
        super(CriticNet, self).__init__()


        self.feature_size = feature_size
        self.action_size = action_size

        self.feature_extractor = feature_extractor

        self.fc1 = nn.Linear(feature_size, hidden1)

        self.fc2 = nn.Linear(hidden1 + self.action_size, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)


    def forward(self, x):

        obs, action = x


        obs = self.feature_extractor(obs)
        obs = F.relu(self.fc1(obs))

        x = torch.cat((obs, action), 1)

        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x.squeeze(1)
