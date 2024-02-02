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
from model import LTC1

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

train_epoches = 10
model = LTC1()
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

loss_list = []

for epoch in range(train_epoches):

    for x, label in tqdm(data_loader, unit='batch', desc='Running epoch {}'.format(epoch)):
        x = x.to(device)
        label = label.to(device)

        label = label.squeeze().long()

        output = model(x)
        loss = criterion(output, label)
        loss_list.append(loss.item())
        #predicted = torch.argmax(output.data, 1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()






