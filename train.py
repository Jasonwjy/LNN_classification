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
from model import *
import csv

# 配置训练参数
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device: ", device)

# ResNet对应的model和criterion

# train_epoches = 25
# model = ResNet(in_channels=1)
# model = model.to(device)
# criterion = nn.BCEWithLogitsLoss()
# model_name = 'ResNet'
# input_size = 32

# ResNet_LTC 对应的model和criterion

# train_epoches = 25
# model = ResNet_LTC(in_channels=1)
# model = model.to(device)
# criterion = nn.BCEWithLogitsLoss()
# model_name = 'ResNet_LTC'
# input_size = 32

# ResNet_CfC 对应的model和criterion

# train_epoches = 100
# model = ResNet_CFC(in_channels=1)
# model = model.to(device)
# criterion = nn.BCEWithLogitsLoss()
# model_name = 'ResNet_CFC'
# input_size = 32

# CNN 对应的model和criterion

# train_epoches = 50
# model = CNN()
# model = model.to(device)
# criterion = nn.CrossEntropyLoss()
# model_name = 'CNN'
# input_size = 128

# CNN_LTC 对应的model和criterion

train_epoches = 50
model = CNN_LTC()
model = model.to(device)
criterion = nn.CrossEntropyLoss()
model_name = 'CNN_LTC'
input_size = 128

print(f'Using model: {model_name}')

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 图像转换为1*28*28的Tensor
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((input_size, input_size)),
    transforms.ToTensor()
])

# 加载数据集
dataset = ImageFolder(root='./data/train', transform=transform)
data_loader = DataLoader(dataset, batch_size=16, shuffle=True)

#加载验证集
val_set = ImageFolder(root='./data/val', transform=transform)
val_loader = DataLoader(val_set, batch_size=16, shuffle=True)

print('Data load completed')

train_loss_list = []
train_acc_list = []


for epoch in range(train_epoches):
    train_loss, train_count = 0, 0
    train_acc = 0
    n_train_correct, n_train_samples = 0, 0
    n_val_correct, n_val_samples = 0, 0

    model.train()
    for x, label in tqdm(data_loader, unit='batch', desc='Running epoch {}'.format(epoch)):
        x = x.to(device)
        if model_name in ['ResNet', 'ResNet_LTC', 'ResNet_CFC']:
            label = label.to(device).to(torch.float)
        else:
            label = label.to(device)

        # 前向传播，计算误差
        output = model(x)
        loss = criterion(output, label)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if model_name in ['ResNet', 'ResNet_LTC', 'ResNet_CFC']:
            y_pred = torch.round(torch.sigmoid(output))
        else:
            y_pred = torch.argmax(output.data, 1)
        train_acc += (y_pred == label).sum().item() / len(label)


    train_loss /= len(data_loader)
    train_acc /= len(data_loader)

    train_loss_list.append(train_loss)
    train_acc_list.append(train_acc)

    print('\ntrain epoch :{}, train_loss :{}, train_acc :{}'.format(epoch, train_loss, train_acc))
    if not os.path.exists('save_model/{}'.format(model_name)):
        os.mkdir('save_model/{}'.format(model_name))
    torch.save(model.state_dict(), 'save_model/{}/model_after_epoch{}.model'.format(model_name, epoch))

with open('CSV/{}_train_data.csv'.format(model_name), mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Epoch', 'Train_loss', 'Train_acc'])

    for i in range(train_epoches):
        writer.writerow([i+1, train_loss_list[i], train_acc_list[i]])

plt.plot(train_loss_list)
plt.savefig('plots/{}_train_loss.png'.format(model_name), format='png', dpi=300)
plt.clf()
plt.plot(train_acc_list)
plt.savefig('plots/{}_train_acc.png'.format(model_name), format='png', dpi=300)
plt.clf()
