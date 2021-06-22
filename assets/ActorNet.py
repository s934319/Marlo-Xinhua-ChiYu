import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorNet(nn.Module):
    def __init__(self, feature_size, feature_extractor, output_size, hidden1=512, hidden2=512):
        super(ActorNet, self).__init__()

        self.feature_size = feature_size
        self.output_size = output_size

        self.feature_extractor = feature_extractor

        self.fc1 = nn.Linear(self.feature_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, self.output_size)



    def forward(self, x):

        x = self.feature_extractor(x)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # x = torch.tanh(self.fc3(x))
        x = F.softmax(self.fc3(x), dim=1)

        return x