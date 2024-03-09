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
test_dataset = ImageFolder(root='train_data', transform=transform)
test_data_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)

# model = CNN()
# model = model.to(device)
# checkpoint = torch.load('save_model/CNN/model_after_epoch9.model')
# model.load_state_dict(checkpoint)

model = LTC1()
model = model.to(device)
checkpoint = torch.load('save_model/LTC/model_after_epoch9.model')
model.load_state_dict(checkpoint)

model.eval()

start_time = time.time()

for x, label in test_data_loader:
    x = x.to(device)
    label = label.to(device)
    pred = model(x)
    end_time = time.time()
    predicted = torch.argmax(pred.data, 1)
    test_acc = ((predicted == label).sum().item()) / 16
    break

time_cost = end_time - start_time
print('单批次推理时间：{:.04f}'.format(time_cost))
print('acc:', test_acc)

total_params = sum(p.numel() for p in model.parameters())
print("模型参数总数：", total_params)