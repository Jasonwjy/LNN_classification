import os
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch
import pandas as pd
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from model import LTC1

# 图像转换为1*28*28的Tensor
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])

# 加载数据集
dataset = ImageFolder(root='train_data', transform=transform)
data_loader = DataLoader(dataset, batch_size=16, shuffle=True)

model = LTC1()

for x, label in data_loader:


    break
