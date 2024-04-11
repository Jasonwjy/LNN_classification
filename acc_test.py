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
from model import LTC1, CNN
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device: ", device)

# 图像转换为1*28*28的Tensor
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])

# 加载数据集
test_dataset = ImageFolder(root='test_data', transform=transform)
test_data_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)

model = LTC1()
model = model.to(device)

acc_list = []

for i in range(25):
    checkpoint = torch.load('save_model/LTC/model_after_epoch{}.model'.format(i))
    model.load_state_dict(checkpoint)

    model.eval()
    n_correct = 0
    n_total = 0

    for x, label in tqdm(test_data_loader, desc='testing epoch {}'.format(i)):
        x = x.to(device)
        label = label.to(device)
        pred = model(x)
        predicted = torch.argmax(pred.data, 1)
        n_correct += ((predicted == label).sum().item())
        n_total += x.size(0)

    epoch_correct = n_correct / n_total
    acc_list.append(epoch_correct)

plt.plot(acc_list)
plt.savefig('LTC_test_acc.png', format='png', dpi=300)
plt.show()