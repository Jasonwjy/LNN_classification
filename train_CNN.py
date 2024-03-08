'''
因为写一个通用的train有点麻烦，索性CNN单独写一个
'''

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
from model import CNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device: ", device)

# 图像转换为1*28*28的Tensor
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])

# 加载数据集
dataset = ImageFolder(root='train_data', transform=transform)
data_loader = DataLoader(dataset, batch_size=16, shuffle=True)

#加载验证集
val_set = ImageFolder(root='val_data', transform=transform)
val_loader = DataLoader(val_set, batch_size=16, shuffle=True)

train_epoches = 10
model = CNN()
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

loss_list = []
train_acc = []
val_acc = []

for epoch in range(train_epoches):
    train_loss, train_count = 0, 0
    n_train_correct, n_train_samples = 0, 0
    n_val_correct, n_val_samples = 0, 0

    #训练
    model.train()
    for x, label in tqdm(data_loader, unit='batch', desc='Running epoch {}'.format(epoch)):
        x = x.to(device)
        label = label.to(device)

        label = label.squeeze().long()

        #前向传播，计算误差
        output = model(x)
        loss = criterion(output, label)

        #记录当前批次的loss,acc
        loss_list.append(loss.item())
        predicted = torch.argmax(output.data, 1)
        n_train_samples += label.size(0)
        n_train_correct += (predicted == label).sum().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    #验证
    model.eval()
    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)
        labels = labels.squeeze().long()

        outputs = model(images)
        loss = criterion(outputs, labels)

        predicted = torch.argmax(outputs.data, 1)

        n_val_samples += labels.size(0)
        n_val_correct += (predicted == labels).sum().item()

    if not os.path.exists('save_model/CNN'):
        os.mkdir('save_model/CNN')
    torch.save(model.state_dict(), 'save_model/CNN/model_after_epoch{}.model'.format(epoch))

    train_acc.append(n_train_correct/n_train_samples)
    val_acc.append(n_val_correct/n_val_samples)
    print('train_acc:', n_train_correct/n_train_samples)
    print('val_acc:', n_val_correct/n_val_samples)

plt.plot(loss_list)
plt.savefig('CNN_loss.png', format='png', dpi=300)
plt.clf()
plt.plot(train_acc)
plt.savefig('CNN_train_acc.png', format='png', dpi=300)
plt.clf()
plt.plot(val_acc)
plt.savefig('CNN_val_acc.png', format='png', dpi=300)
plt.clf()







