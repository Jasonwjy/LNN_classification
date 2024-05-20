import os
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from model import *  # 确保这个导入正确
from sklearn.metrics import precision_score, recall_score, f1_score
import csv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device: ", device)

epochs = 25
input_size = 32
model_name = 'ResNet_CFC'

# 初始化模型，并将其移至设定的设备上
model = ResNet_CFC()
model = model.to(device)

# 定义转换，用于测试集的图片预处理
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((input_size, input_size)),
    transforms.ToTensor()
])

# 加载测试数据集
test_dataset = ImageFolder(root='data/test', transform=transform)
test_data_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)



# 初始化存储指标的列表
acc_list, f1_list, precision_list, recall_list = [], [], [], []

# 测试循环
for i in range(epochs):
    checkpoint = torch.load('save_model/{}/model_after_epoch{}.model'.format(model_name, i))
    model.load_state_dict(checkpoint)
    model.eval()
    n_correct, n_total = 0, 0
    y_true, y_pred = [], []

    for x, label in tqdm(test_data_loader, desc='Testing epoch {}'.format(i), unit='batch'):
        x, label = x.to(device), label.to(device)
        pred = model(x)
        if model_name in ['ResNet', 'ResNet_LTC', 'ResNet_CFC']:
            predicted = pred.data
        else:
            predicted = torch.argmax(pred.data, 1)
        n_correct += (predicted == label).sum().item()
        n_total += x.size(0)
        y_true.extend(label.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

    accuracy = n_correct / n_total
    acc_list.append(accuracy)
    y_pred = np.array(y_pred)
    y_pred_int = (y_pred > 0).astype(int)
    precision = precision_score(y_true, y_pred_int, zero_division=0)
    recall = recall_score(y_true, y_pred_int, zero_division=0)
    f1 = f1_score(y_true, y_pred_int, zero_division=0)
    precision_list.append(precision)
    recall_list.append(recall)
    f1_list.append(f1)
    print('\nTesting epoch: {}, Accuracy: {}, Precision: {}, Recall: {}, F1 Score: {}\n'.format(i, accuracy, precision, recall, f1))

# 存储指标到 CSV 文件
with open('CSV/{}_test_result.csv'.format(model_name), mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Epoch', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])
    for i in range(epochs):
        writer.writerow([i+1, acc_list[i], precision_list[i], recall_list[i], f1_list[i]])

# 绘制指标图表
plt.clf()
plt.plot(acc_list)
plt.title("Accuracy Over Epochs")
plt.ylim(0.0, 1.0)
plt.savefig('plots/{}_test_acc.png'.format(model_name), format='png', dpi=300)
plt.clf()
plt.plot(precision_list)
plt.title("Precision Over Epochs")
plt.ylim(0.0, 1.0)
plt.savefig('plots/{}_test_precision.png'.format(model_name), format='png', dpi=300)
plt.clf()
plt.plot(recall_list)
plt.title("Recall Over Epochs")
plt.ylim(0.0, 1.0)
plt.savefig('plots/{}_test_recall.png'.format(model_name), format='png', dpi=300)
plt.clf()
plt.plot(f1_list)
plt.title("F1 Score Over Epochs")
plt.ylim(0.0, 1.0)
plt.savefig('plots/{}_test_F1.png'.format(model_name), format='png', dpi=300)
plt.show()
