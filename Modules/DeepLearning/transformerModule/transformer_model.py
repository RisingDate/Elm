import numpy as np
import time
import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
import joblib
from sklearn.preprocessing import LabelEncoder
from dataProcess import data_process, CustomDataset
from model import XTransformer


class LogCoshLoss(nn.Module):
    def forward(self, y_pred, y_true):
        return torch.mean(torch.log(torch.cosh(y_pred - y_true)))


if __name__ == '__main__':
    path = '../../../Dataset/A/train_data.txt'
    train_data = data_process(path)

    features = ['site_id', 'statistical_duration', 'publish_weekday', 'gender', 'age', 'fans_cnt', 'coin_cnt',
                'video_cnt', 'post_type', 'city_level', 'authority_popularity', 'fans_video_ratio', 'avg_coin_per_video',
                'avg_fans_per_video']
    x_train = train_data[features].values
    y_train = train_data['interaction_cnt'].values
    y_train = np.log(y_train + 1)

    # 数据标准化
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    joblib.dump(scaler, '../models/tf-scaler5.pkl')

    # 初始化模型
    input_size = x_train.shape[1]
    model = XTransformer(input_dim=input_size, dim=64, depth=4, heads=4)
    # model = EnhancedInteractionPredictor(input_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建数据集和数据加载器
    x_train = torch.tensor(x_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    train_dataset = CustomDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # 定义损失函数和优化器
    criterion = nn.HuberLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    # criterion = LogCoshLoss()
    # optimizer = optim.NAdam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)  # 学习率调度器
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
            scheduler.step()

            # 转换回原始尺度并计算绝对值之差
            original_labels = torch.exp(labels)  # y = exp(log(y))
            original_outputs = torch.exp(outputs)  # pred = exp(log(pred))
            abs_diff = torch.abs(original_labels - original_outputs)

            score += abs_diff.sum().item()
            running_loss += loss.item()

        epoch_end_time = time.time()
        # 计算评分
        print(f'Epoch {epoch + 1}/{num_epochs}, '
              f'xTrain Len {len(x_train)}, '
              f'Loss: {running_loss / len(train_loader)}, '
              f'Score: {score / len(x_train)}, '
              f'Running Time: {epoch_end_time - epoch_strat_time}')

    # 保存模型
    torch.save(model, '../models/tf-model5.pth')
