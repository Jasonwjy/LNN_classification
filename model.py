#pc commit test
import torch
import torch.nn as nn
import torch.nn.functional as F

from ncps.torch import CfC, LTC
from ncps.wirings import AutoNCP, NCP
import matplotlib.pyplot as plt
import seaborn as sns

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LTC(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LTC, self).__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_classes = num_classes

        #CNN部分，负责图像特征提取
        self.conv1 = nn.Conv2d(1, 16, 3)  # in channels, output channels, kernel size
        self.conv2 = nn.Conv2d(16, 32, 3, padding=2, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 5, padding=2, stride=2)
        self.conv4 = nn.Conv2d(64, 128, 5, padding=2, stride=2)
        self.bn4 = nn.BatchNorm2d(128)

    def forward(self, x):
        # x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        # x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))

        x = F.relu(self.conv1(x))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(F.relu(self.bn4(self.conv4(x))), (2, 2))

        return x