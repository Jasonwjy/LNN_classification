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
from sklearn.metrics import f1_score


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device: ", device)

epoches = 25
input_size = 32
model_name = 'ResNet_LTC'

# 图像转换为1*28*28的Tensor
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((input_size, input_size)),
    transforms.ToTensor()
])

# 加载数据集
test_dataset = ImageFolder(root='data/test', transform=transform)
test_data_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)

model = ResNet_LTC(in_channels=1)
# model = CNN()
model = model.to(device)

acc_list = []
f1_list = []

for i in range(epoches):
    checkpoint = torch.load('save_model/{}/model_after_epoch{}.model'.format(model_name, i))
    model.load_state_dict(checkpoint)

    model.eval()
    n_correct = 0
    n_total = 0
    y_true = []
    y_pred = []

    for x, label in tqdm(test_data_loader, desc='testing epoch {}'.format(i), unit='batch'):
        x = x.to(device)
        label = label.to(device)
        pred = model(x)
        # CNN 和 LTC1 使用这条
        if model_name in ['ResNet', 'ResNet_LTC']:
            predicted = pred.data
        else:
            predicted = torch.argmax(pred.data, 1)
        predicted = (predicted > 0)
        n_correct += ((predicted == label).sum().item())
        n_total += x.size(0)
        y_true.extend(label.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

    epoch_correct = n_correct / n_total
    acc_list.append(epoch_correct)
    y_pred = np.array(y_pred)
    y_pred_int = (y_pred > 0).astype(int)
    f1 = f1_score(y_true, y_pred_int)
    f1_list.append(f1)
    print('\ntesting epoch: {}, accuracy: {}, F1_score: {}'.format(i, epoch_correct, f1))

plt.plot(acc_list)
plt.ylim(0.0, 1.0)
plt.savefig('plots/{}_test_acc.png'.format(model_name), format='png', dpi=300)
plt.clf()
plt.plot(f1_list)
plt.ylim(0.0, 1.0)
plt.savefig('plots/{}_test_F1.png'.format(model_name), format='png', dpi=300)
plt.show()