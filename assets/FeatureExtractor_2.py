import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureExtractor_2(nn.Module):
    def __init__(self, input_shape, feature_size):
        super(FeatureExtractor_2, self).__init__()

        # C, H, W
        self.input_shape = input_shape
        self.feature_size = feature_size

        

        # self.conv1 = nn.Conv2d(input_shape[0], 32, 5, stride=1, padding=2)
        # self.maxp1 = nn.MaxPool2d(2, 2)
        # self.conv2 = nn.Conv2d(32, 32, 5, stride=1, padding=1)
        # self.maxp2 = nn.MaxPool2d(2, 2)
        # self.conv3 = nn.Conv2d(32, 64, 4, stride=1, padding=1)
        # self.maxp3 = nn.MaxPool2d(2, 2)
        # self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        # self.maxp4 = nn.MaxPool2d(2, 2)

        
        self.conv1 = nn.Conv2d(input_shape[0], 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)

        self.fc1 = nn.Linear(self.conv_size(), self.feature_size)


    def conv_size(self):
        with torch.no_grad():
            return self.conv3(self.conv2(self.conv1(torch.zeros(1, *self.input_shape)))).view(1, -1).size(1)


    def forward(self, x):
        # x = F.relu(self.maxp1(self.conv1(x)))
        # x = F.relu(self.maxp2(self.conv2(x)))
        # x = F.relu(self.maxp3(self.conv3(x)))
        # x = F.relu(self.maxp4(self.conv4(x)))

        x = (x - self.obs_mean) / self.obs_std

        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))

        x = x.view(x.size(0), -1)

        x = self.fc1(x)

        return x

    def set(self, obs_mean, obs_std):
        self.obs_mean = obs_mean
        self.obs_std = obs_std