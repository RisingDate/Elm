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
    features = ['fans_cnt', 'coin_cnt']
    x_train = train_data[features].values
    y_train = train_data['interaction_cnt'].values
    y_train = np.log(y_train+1)

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
    criterion = nn.HuberLoss()  # 替换为更鲁棒的损失函数
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

    # 训练模型
    num_epochs = 50
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        epoch_strat_time = time.time()
        running_loss = 0.0
        score = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 转换回原始尺度并计算绝对值之差
            original_labels = torch.exp(labels)  # y = exp(log(y))
            original_outputs = torch.exp(outputs)  # pred = exp(log(pred))
            abs_diff = torch.abs(original_labels - original_outputs)

            score += abs_diff.sum().item()
            running_loss += loss.item()

        epoch_end_time = time.time()
        # 计算评分
        print(f'Epoch {epoch + 1}/{num_epochs}, '
              f'Loss: {running_loss / len(train_loader)}, '
              f'Score: {score}, '
              f'Running Time: {epoch_end_time - epoch_strat_time}')

    # 保存模型
    torch.save(model, './models/model-with-2-feature.pth')
