import numpy as np
import time
import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
import joblib
from sklearn.preprocessing import LabelEncoder
from dataProcess import data_process, TabDataset
from model import TabTransformer


class LogCoshLoss(nn.Module):
    def forward(self, y_pred, y_true):
        return torch.mean(torch.log(torch.cosh(y_pred - y_true)))


params = {
    'train_data_path': '../../../Dataset/A/train_data.txt',
    'label_encoders_save_path': '../models/label_encoders9.pkl',
    'scaler_save_path': '../models/tf-scaler9.pkl',
    'model_save_path': '../models/tf-model9.pth'
}
if __name__ == '__main__':
    path = params['train_data_path']
    train_data = data_process(path)

    # 定义数值特征与分类特征
    categorical_features = ['site_id', 'gender', 'post_type', 'city_level']
    numeric_features = ['statistical_duration', 'publish_weekday', 'age', 'fans_cnt', 'coin_cnt',
                        'video_cnt', 'authority_popularity', 'fans_video_ratio', 'avg_coin_per_video',
                        'avg_fans_per_video', 'site_post', 'site_age_group', 'site_city']

    # 标签
    y_train = np.log(train_data['interaction_cnt'].values + 1)

    # 编码分类变量
    label_encoders = {}
    x_categorical = {}
    categorical_info = {}

    for col in categorical_features:
        le = LabelEncoder()
        train_data[col] = le.fit_transform(train_data[col])
        x_categorical[col] = train_data[col].values
        label_encoders[col] = le
        categorical_info[col] = len(le.classes_)

    # 数值特征标准化
    x_numeric = train_data[numeric_features].values
    scaler = StandardScaler()
    x_numeric = scaler.fit_transform(x_numeric)

    # 保存预处理器
    joblib.dump(scaler, params['scaler_save_path'])
    joblib.dump(label_encoders, params['label_encoders_save_path'])

    # 数据加载器
    dataset = TabDataset(x_numeric, x_categorical, y_train, categorical_info)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # 模型初始化
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TabTransformer(num_numeric_features=len(numeric_features),
                           categorical_info=categorical_info,
                           embed_dim=32, dim=64, depth=4, heads=4).to(device)

    criterion = LogCoshLoss()
    optimizer = optim.NAdam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)

    # 模型训练
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        epoch_start = time.time()
        total_loss = 0.0
        total_score = 0.0

        for x_cat_batch, x_num_batch, y_batch in dataloader:
            for k in x_cat_batch:
                x_cat_batch[k] = x_cat_batch[k].to(device)
            x_num_batch = x_num_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            preds = model(x_num_batch, x_cat_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            scheduler.step()

            # 反变换对比
            true = torch.exp(y_batch)
            pred = torch.exp(preds)
            total_score += torch.abs(true - pred).sum().item()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs} | "
              f"Loss: {total_loss / len(dataloader):.4f} | "
              f"Score: {total_score / len(dataset):.4f} | "
              f"Time: {time.time() - epoch_start:.2f}s")

    # 保存模型
    torch.save(model.state_dict(), params['model_save_path'])
