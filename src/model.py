from torch import nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square conv kernel
        self.conv1 = nn.Conv2d(3, 8, 3)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, 3)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, 5, stride = 2, padding = 1)
        self.bn3 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(32 * 13 * 13, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.bn1(F.max_pool2d(F.relu(self.conv1(x)), 2))
        x = self.bn2(F.max_pool2d(F.relu(self.conv2(x)), 2))
        x = self.bn3(F.max_pool2d(F.relu(self.conv3(x)), 2))
        x = x.view(x.shape[0], -1)
        x = self.bn4(F.relu(self.fc1(x)))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
