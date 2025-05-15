import numpy as np
import time
import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
import joblib
from sklearn.preprocessing import LabelEncoder
from dataProcess import data_process
from model import XTransformerWithEmbedding  # 注意模型名


class LogCoshLoss(nn.Module):
    def forward(self, y_pred, y_true):
        return torch.mean(torch.log(torch.cosh(y_pred - y_true)))


class CustomDatasetWithCat(Dataset):
    def __init__(self, x_numeric, x_cat_dict, y):
        self.x_numeric = x_numeric
        self.x_cat_dict = x_cat_dict
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        cat_inputs = {k: v[idx] for k, v in self.x_cat_dict.items()}
        return self.x_numeric[idx], cat_inputs, self.y[idx]


params = {
    'train_data_path': '../../../Dataset/A/train_data.txt',
    'label_encoders_save_path': '../models/label_encoders8.pkl',
    'scaler_save_path': '../models/tf-scaler8-with-text.pkl',
    'model_save_path': '../models/tf-model8-with-text.pth'
}
if __name__ == '__main__':
    path = params['train_data_path']
    train_data = data_process(path)

    # 处理分类特征
    str_features = ['user_site', 'user_post', 'uid']
    label_encoders = {}
    for col in str_features:
        le = LabelEncoder()
        train_data[col] = le.fit_transform(train_data[col])
        label_encoders[col] = le
    joblib.dump(label_encoders, params['label_encoders_save_path'])
    categorical_info = {col: train_data[col].nunique() for col in str_features}

    # 处理数值特征
    numeric_features = ['site_id', 'statistical_duration', 'publish_weekday', 'gender', 'age', 'fans_cnt', 'coin_cnt',
                        'video_cnt', 'post_type', 'city_level', 'authority_popularity', 'fans_video_ratio',
                        'avg_coin_per_video', 'avg_fans_per_video', 'site_post', 'site_age_group', 'site_city']
    x_numeric = train_data[numeric_features].values
    y_train = train_data['interaction_cnt'].values
    y_train = np.log(y_train + 1)

    # 标准化
    scaler = StandardScaler()
    x_numeric = scaler.fit_transform(x_numeric)
    joblib.dump(scaler, params['scaler_save_path'])

    # 转为 Tensor
    x_numeric = torch.tensor(x_numeric, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    x_cat_dict = {col: torch.tensor(train_data[col].values, dtype=torch.long) for col in str_features}

    # 初始化模型
    model = XTransformerWithEmbedding(
        num_numeric_features=len(numeric_features),
        categorical_info=categorical_info,
        embed_dim=8,
        dim=64,
        depth=4,
        heads=4,
        dropout=0.1
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    x_numeric = x_numeric.to(device)
    y_train = y_train.to(device)
    x_cat_dict = {k: v.to(device) for k, v in x_cat_dict.items()}

    train_dataset = CustomDatasetWithCat(x_numeric, x_cat_dict, y_train)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # 损失函数与优化器
    criterion = nn.HuberLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)

    # 训练过程
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        epoch_strat_time = time.time()
        running_loss = 0.0
        score = 0.0
        for x_num, x_cat, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(x_num, x_cat).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            # 还原回原始尺度计算分数
            original_labels = torch.exp(labels)
            original_outputs = torch.exp(outputs)
            abs_diff = torch.abs(original_labels - original_outputs)

            score += abs_diff.sum().item()
            running_loss += loss.item()

        epoch_end_time = time.time()
        print(f'Epoch {epoch + 1}/{num_epochs}, '
              f'xTrain Len {len(x_numeric)}, '
              f'Loss: {running_loss / len(train_loader):.4f}, '
              f'Score: {score / len(x_numeric):.4f}, '
              f'Running Time: {epoch_end_time - epoch_strat_time:.2f}s')

    torch.save(model, params['model_save_path'])
