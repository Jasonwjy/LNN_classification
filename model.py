import torch
import torch.nn as nn
import torch.nn.functional as F

from ncps.torch import CfC, LTC
from ncps.wirings import AutoNCP, NCP
import matplotlib.pyplot as plt
import seaborn as sns

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 这个模型适用的是28*28的输入尺寸，但是我认为这样的输入尺寸可能有点模糊
# 后面可能会测试更大的输入尺寸，这样的话模型的参数要重新改一下
class LTC1(nn.Module):
    def __init__(self):
        super(LTC1, self).__init__()
        self.ltc_input_size = 16
        self.ltc_hidden_size = 19
        self.num_classes = 2
        self.seq_length = 32

        # 这个模型的头部好像有点问题，换一个
        #CNN部分，负责图像特征提取
        # self.conv1 = nn.Conv2d(1, 16, 3)  # in channels, output channels, kernel size
        # self.conv2 = nn.Conv2d(16, 32, 3, padding=2, stride=2)
        # self.bn2 = nn.BatchNorm2d(32)
        # self.conv3 = nn.Conv2d(32, 64, 5, padding=2, stride=2)
        # self.conv4 = nn.Conv2d(64, 128, 5, padding=2, stride=2)
        # self.bn4 = nn.BatchNorm2d(128)

        self.conv1 = nn.Conv2d(1, 6, 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)

        wiring = AutoNCP(self.ltc_hidden_size, self.num_classes)
        self.rnn = LTC(self.ltc_input_size, wiring)

    def forward(self, x):
        # x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        # x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))

        # x = F.relu(self.conv1(x))
        # x = F.relu(self.bn2(self.conv2(x)))
        # #x = F.max_pool2d(x, (2, 2))
        # x = F.relu(self.conv3(x))
        # x = F.max_pool2d(F.relu(self.bn4(self.conv4(x))), (2, 2))
        # print(x.shape)

        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        print(x.shape)
        x = x.view(x.shape[0], -1)

        x = x.view(-1, self.seq_length, self.ltc_input_size)
        print(x.shape)
        h0 = torch.zeros(x.size(0), self.ltc_hidden_size).to(device)
        out, _ = self.rnn(x, h0)
        out = out[:, -1, :]

        return out

# 这个模型烂了
class CNN(nn.Module):
    def __init__(self):
        img_size = 128
        num_class = 2
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            # default parameter：nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.conv8 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )

        self.conv9 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.conv10 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.conv11 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.conv12 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
        self.conv13 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.conv14 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.conv15 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.conv16 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )

        self.fc17 = nn.Sequential(
            nn.Linear(int(512 * img_size * img_size / 32 / 32), 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5)  # 默认就是0.5
        )

        self.fc18 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5)
        )

        self.fc19 = nn.Sequential(
            nn.Linear(4096, num_class)
        )

        self.conv_list = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6, self.conv7,
                          self.conv8, self.conv9, self.conv10, self.conv11, self.conv12, self.conv13, self.conv14,
                          self.conv15, self.conv16]

        self.fc_list = [self.fc17, self.fc18, self.fc19]

    def forward(self, x):
        for conv in self.conv_list:  # 16 CONV
            x = conv(x)
        output = x.view(x.size()[0], -1)
        for fc in self.fc_list:  # 3 FC
            output = fc(output)
        return output


class CNN_LTC(nn.Module):
    def __init__(self):
        img_size = 128
        num_class = 2
        super(CNN_LTC, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            # default parameter：nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.conv8 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )

        self.conv9 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.conv10 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.conv11 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.conv12 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
        self.conv13 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.conv14 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.conv15 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.conv16 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )

        self.conv_list = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6, self.conv7,
                          self.conv8, self.conv9, self.conv10, self.conv11, self.conv12, self.conv13, self.conv14,
                          self.conv15, self.conv16]

        self.wiring = AutoNCP(28, 2)
        self.rnn = LTC(64, self.wiring)

    def forward(self, x):
        for conv in self.conv_list:  # 16 CONV
            x = conv(x)
        # output = x.view(x.size()[0], -1)
        output = x.view(-1, 128, 64)
        h0 = torch.zeros(output.size(0), 28).to(device)  # hidden_size = 28,对应AutoNCP的units

        x, _ = self.rnn(output, h0)
        x = x[:, -1, :]
        return x.squeeze()

# kaggle上找了一个resnet的新模型，准确率还可以
# https://www.kaggle.com/code/kannapat/pneumonia-detection-with-pytorch-81-acc
# 模型要求 1*32*32 的输入
class ResNet(nn.Module):
    def conv_block(self, in_channels, out_channels, pool=False):
      layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)]
      if pool: layers.append(nn.MaxPool2d(2))
      return nn.Sequential(*layers)

    def __init__(self, in_channels=1):
        super().__init__()

        self.conv1 = self.conv_block(in_channels, 16)
        self.conv2 = self.conv_block(16, 32, pool=True)
        self.res1 = nn.Sequential(self.conv_block(32,32), self.conv_block(32,32))

        self.conv3 = self.conv_block(32, 64, pool=True)
        self.conv4 = self.conv_block(64, 128, pool=True)
        self.res2 = nn.Sequential(self.conv_block(128,128), self.conv_block(128,128))


        self.classifier = nn.Sequential(nn.MaxPool2d(3),
                                        nn.Flatten(),
                                        nn.Dropout(p=0.5),
                                        nn.Linear(in_features=128,out_features=1))


    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out.squeeze()

class ResNet_LTC(nn.Module):
    def conv_block(self, in_channels, out_channels, pool=False):
      layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)]
      if pool: layers.append(nn.MaxPool2d(2))
      return nn.Sequential(*layers)

    def __init__(self, in_channels=1):
        super().__init__()
        self.units = 28

        self.conv1 = self.conv_block(in_channels, 16)
        self.conv2 = self.conv_block(16, 32, pool=True)
        self.res1 = nn.Sequential(self.conv_block(32,32), self.conv_block(32,32))

        self.conv3 = self.conv_block(32, 64, pool=True)
        self.conv4 = self.conv_block(64, 128, pool=True)
        self.res2 = nn.Sequential(self.conv_block(128,128), self.conv_block(128,128))

        self.pool = nn.MaxPool2d(3)

        self.wiring = AutoNCP(self.units, 1)
        self.rnn = LTC(16, self.wiring)

        # 这行代码是用来绘制模型内部连接结构的
        # make_wiring_diagram(self.wiring, "kamada", model_name='LTC_16_NCP')


        # self.classifier = nn.Sequential(nn.MaxPool2d(3),
        #                                 nn.Flatten(),
        #                                 nn.Dropout(p=0.5),
        #                                 nn.Linear(in_features=128,out_features=1))

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.pool(out)
        out = out.view(-1, 8, 16)       # seq_length = 8, input_size = 16
        h0 = torch.zeros(out.size(0), self.units).to(device)  # hidden_size = 28,对应AutoNCP的units

        x, _ = self.rnn(out, h0)
        x = x[:, -1, :]
        return x.squeeze()

class ResNet_CFC(nn.Module):
    def conv_block(self, in_channels, out_channels, pool=False):
      layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)]
      if pool: layers.append(nn.MaxPool2d(2))
      return nn.Sequential(*layers)

    def __init__(self, in_channels=1):
        super().__init__()
        self.units = 28

        self.conv1 = self.conv_block(in_channels, 16)
        self.conv2 = self.conv_block(16, 32, pool=True)
        self.res1 = nn.Sequential(self.conv_block(32,32), self.conv_block(32,32))

        self.conv3 = self.conv_block(32, 64, pool=True)
        self.conv4 = self.conv_block(64, 128, pool=True)
        self.res2 = nn.Sequential(self.conv_block(128,128), self.conv_block(128,128))

        self.pool = nn.MaxPool2d(3)

        self.wiring = AutoNCP(self.units, 1)
        self.rnn = CfC(16, self.wiring)
        # self.rnn = CfC(16, units=self.units, )

        # 这行代码是用来绘制模型内部连接结构的
        # make_wiring_diagram(self.wiring, "kamada")


        # self.classifier = nn.Sequential(nn.MaxPool2d(3),
        #                                 nn.Flatten(),
        #                                 nn.Dropout(p=0.5),
        #                                 nn.Linear(in_features=128,out_features=1))

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.pool(out)
        out = out.view(-1, 8, 16)       # seq_length = 8, input_size = 16
        h0 = torch.zeros(out.size(0), self.units).to(device)  # hidden_size = 28,对应AutoNCP的units

        x, _ = self.rnn(out, h0)
        x = x[:, -1, :]
        return x.squeeze()

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

def make_wiring_diagram(wiring, layout, model_name):
    sns.set_style("white")
    plt.figure(figsize=(12, 12))
    legend_handles = wiring.draw_graph(layout=layout,neuron_colors={"command": "tab:cyan"})
    plt.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(1, 1))
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.savefig('plots/{}.png'.format(model_name), format='png', dpi=300)