import torch
import torch.nn as nn
import torch.nn.functional as F

from ncps.torch import CfC, LTC
from ncps.wirings import AutoNCP, NCP
import matplotlib.pyplot as plt
import seaborn as sns

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#这个模型适用的是28*28的输入尺寸，但是我认为这样的输入尺寸可能有点模糊
#后面可能会测试更大的输入尺寸，这样的话模型的参数要重新改一下
class LTC1(nn.Module):
    def __init__(self):
        super(LTC1, self).__init__()
        self.ltc_input_size = 16
        self.ltc_hidden_size = 19
        self.num_classes = 2
        self.seq_length = 32

        #CNN部分，负责图像特征提取
        self.conv1 = nn.Conv2d(1, 16, 3)  # in channels, output channels, kernel size
        self.conv2 = nn.Conv2d(16, 32, 3, padding=2, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 5, padding=2, stride=2)
        self.conv4 = nn.Conv2d(64, 128, 5, padding=2, stride=2)
        self.bn4 = nn.BatchNorm2d(128)

        wiring = AutoNCP(self.ltc_hidden_size, self.num_classes)
        self.rnn = LTC(self.ltc_input_size, wiring)

    def forward(self, x):
        # x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        # x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))

        x = F.relu(self.conv1(x))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(F.relu(self.bn4(self.conv4(x))), (2, 2))

        x = x.view(-1, self.seq_length, self.ltc_input_size)
        h0 = torch.zeros(x.size(0), self.ltc_hidden_size).to(device)
        out, _ = self.rnn(x, h0)
        out = out[:, -1, :]

        return out

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3)  # in channels, output channels, kernel size
        self.conv2 = nn.Conv2d(16, 32, 3, padding=2, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 5, padding=2, stride=2)
        self.conv4 = nn.Conv2d(64, 128, 5, padding=2, stride=2)
        self.bn4 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(512, 256) # 输入特征大小为 128 * 2 * 2，输出特征大小为 512
        self.fc2 = nn.Linear(256, 2) # 输入特征大小为 512，输出特征大小为 256

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(F.relu(self.bn4(self.conv4(x))), (2, 2))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

#这个LNN是原作者（LNN_cancer_classification项目）里用的模型
class LNN(nn.Module):
    def __init__(self, ncp_input_size, hidden_size, num_classes, sequence_length):
        super(LNN, self).__init__()

        self.hidden_size = hidden_size
        self.ncp_input_size = ncp_input_size
        self.sequence_length = sequence_length

        ### CNN HEAD
        self.conv1 = nn.Conv2d(1, 16, 3)  # in channels, output channels, kernel size
        self.conv2 = nn.Conv2d(16, 32, 3, padding=2, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 5, padding=2, stride=2)
        self.conv4 = nn.Conv2d(64, 128, 5, padding=2, stride=2)
        self.bn4 = nn.BatchNorm2d(128)

        ### DESIGNED NCP architecture
        wiring = AutoNCP(hidden_size, num_classes)  # 234,034 parameters

        # wiring = NCP(
        #     inter_neurons=13,  # Number of inter neurons
        #     command_neurons=4,  # Number of command neurons
        #     motor_neurons=2,  # Number of motor neurons
        #     sensory_fanout=4,  # How many outgoing synapses has each sensory neuron
        #     inter_fanout=2,  # How many outgoing synapses has each inter neuron
        #     recurrent_command_synapses=3,  # Now many recurrent synapses are in the
        #     # command neuron layer
        #     motor_fanin=4,  # How many incomming syanpses has each motor neuron
        # )
        self.rnn = CfC(ncp_input_size, wiring)

        make_wiring_diagram(wiring, "kamada")

        ### Fully connected NCP architecture
        # self.rnn = CfC(ncp_input_size, hidden_size, proj_size = num_classes, batch_first = True) # input shape -> batch_size, seq len, feature_size (input size)  . Batch_first just means we need that batch dim present

    def forward(self, x):
        # x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        # x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))

        x = F.relu(self.conv1(x))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(F.relu(self.bn4(self.conv4(x))), (2, 2))

        ## RNN MODE
        x = x.view(-1, self.sequence_length, self.ncp_input_size)
        h0 = torch.zeros(x.size(0), self.hidden_size).to(device)

        out, _ = self.rnn(x, h0)
        out = out[:, -1,
              :]  # we have 28 outputs since each part of sequence generates an output. for classification, we only want the last one
        return out

def make_wiring_diagram(wiring, layout):
    sns.set_style("white")
    plt.figure(figsize=(12, 12))
    legend_handles = wiring.draw_graph(layout=layout,neuron_colors={"command": "tab:cyan"})
    plt.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(1, 1))
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.show()