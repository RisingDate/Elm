import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, r2_score


# 定义全连接神经网络模型
class SimpleNet(nn.Module):
    def __init__(self, input_size):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out



# # 评估模型
# model.eval()
# y_pred = []
# with torch.no_grad():
#     for inputs, _ in test_loader:
#         inputs = inputs.to(device)
#         outputs = model(inputs)
#         y_pred.extend(outputs.cpu().numpy().flatten())
#
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)
# print(f'Mean Squared Error: {mse}')
# print(f'R-squared: {r2}')
#
#
#
# # 保存整个模型
# torch.save(model, './models/model1.pth')
# # 加载整个模型
# loaded_model = torch.load('full_model.pth')
# loaded_model.eval()  # 设置为评估模式