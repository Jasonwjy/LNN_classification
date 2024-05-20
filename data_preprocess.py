import os
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import csv

# 配置训练参数
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device: ", device)

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
        # input_tensor = x
        # input_tensor = input_tensor.to('cpu')
        # input_tensor = input_tensor.permute(0, 2, 3, 1)
        # input_tensor = input_tensor.detach().numpy()
        # plt.imshow(input_tensor[0, :, :, 0], cmap='gray')
        # plt.show()
        # plt.clf()
        for i, conv in enumerate(self.conv_list):  # 16 CONV
            x = conv(x)
            print(x.shape)
            # if i == 0 or i == 1:
            #     input_tensor = x
            #     input_tensor = input_tensor.to('cpu')
            #     input_tensor = input_tensor.permute(0, 2, 3, 1)
            #     input_tensor = input_tensor.detach().numpy()
            #     plot_images(input_tensor)
        output = x.view(x.size()[0], -1)
        for fc in self.fc_list:  # 3 FC
            output = fc(output)
        return output

def plot_images(tensor, num_images=8, cols=4):
    print(tensor.shape)
    rows = num_images // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    for i, ax in enumerate(axes.flatten()):
        if i < num_images:
            ax.imshow(tensor[0, :, :, i], cmap='gray')  # 将图像绘制为灰度图
            ax.axis('off')
    plt.tight_layout()
    plt.show()

# CNN 对应的model和criterion

train_epoches = 50
model = CNN()
model = model.to(device)
criterion = nn.CrossEntropyLoss()
model_name = 'CNN'
input_size = 128


optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 图像转换为1*28*28的Tensor
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((input_size, input_size)),
    transforms.ToTensor()
])

# 加载数据集
dataset = ImageFolder(root='./data/train', transform=transform)
data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

#加载验证集
val_set = ImageFolder(root='./data/val', transform=transform)
val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

print('Data load completed')

train_loss_list = []
train_acc_list = []


model.train()
for x, label in tqdm(data_loader, unit='batch', desc='Running epoch {}'.format(1)):
    x = x.to(device)
    if model_name in ['ResNet', 'ResNet_LTC', 'ResNet_CFC']:
        label = label.to(device).to(torch.float)
    else:
        label = label.to(device)
    output = model(x)
    print(output)

    break




