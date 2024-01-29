import os
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch
import pandas as pd
from tqdm import tqdm

# 图像转换为1*28*28的Tensor
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])

# 加载数据集
dataset = ImageFolder(root='Data/train', transform=transform)

# 数据加载器
data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

# 初始化 NORMAL, VIRUS, BACTERIA 的 Tensor 列表
normal_data = []
virus_bacteria_data = []

# 遍历数据集
for img, label in tqdm(data_loader, desc='Processing Images', unit='image'):
    # 将图片添加到相应的列表中
    if label == 1:  # NORMAL
        normal_data.append(img.view(1, -1))
    elif label == 0 or label == 2:  # VIRUS or BACTERIA
        virus_bacteria_data.append(img.view(1, -1))

# 合并 VIRUS 和 BACTERIA 的 Tensor
virus_bacteria_tensor = torch.cat(virus_bacteria_data, dim=0)

# 将 NORMAL 的 Tensor 转为 DataFrame，并保存为 CSV
normal_df = pd.DataFrame(torch.cat(normal_data, dim=0).numpy())
normal_df.to_csv('normal_data.csv', index=False)

# 将 VIRUS 和 BACTERIA 的 Tensor 转为 DataFrame，并保存为 CSV
virus_bacteria_df = pd.DataFrame(virus_bacteria_tensor.numpy())
virus_bacteria_df.to_csv('virus_bacteria_data.csv', index=False)
