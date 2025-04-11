import numpy as np
import time
import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from dataProcess import data_process, CustomDataset
from model import SimpleNet, InteractionPredictor

if __name__ == '__main__':
    path = '../../Dataset/A/train.txt'
    train_data = data_process(path)
    # 选择特征和目标变量
    # features = ['site_id', 'statistical_duration', 'gender', 'age', 'fans_cnt', 'coin_cnt', 'post_type']  # 替换为实际的特征列名
    features = ['site_id', 'statistical_duration', 'fans_cnt', 'coin_cnt']
    x_train = train_data[features].values
    y_train = train_data['interaction_cnt'].values

    # 数据标准化
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)

    # 初始化模型
    input_size = x_train.shape[1]
    model = InteractionPredictor(input_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    # 创建数据集和数据加载器
    x_train = torch.tensor(x_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    train_dataset = CustomDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # 定义损失函数和优化器
    criterion = nn.SmoothL1Loss()  # 替换为更鲁棒的损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    # 训练模型
    num_epochs = 50
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        epoch_strat_time = time.time()
        running_loss = 0.0
        all_predictions = []
        all_labels = []
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            # 收集预测结果和实际标签
            all_predictions.append(outputs.detach())
            all_labels.append(labels.detach())

        epoch_end_time = time.time()
        # 计算评分
        all_predictions = torch.cat(all_predictions, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        score = torch.sum(torch.abs(all_predictions - all_labels)).item()
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}, Score: {score}, Running Time: {epoch_end_time - epoch_strat_time}')
