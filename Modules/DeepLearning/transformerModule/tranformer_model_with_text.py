import numpy as np
import time
import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
import joblib
from sklearn.preprocessing import LabelEncoder
from dataProcess import data_process, TabDataset
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
    'label_encoders_save_path': '../models/label_encoders10.pkl',
    'scaler_save_path': '../models/tf-scaler10-with-text.pkl',
    'model_save_path': '../models/tf-model10-with-text.pth'
}
if __name__ == '__main__':
    full_data = data_process(params['train_data_path'])

    # 处理 uid 稀疏问题
    uid_counts = full_data['uid'].value_counts()
    top_uids = uid_counts[uid_counts > 5].index
    full_data['uid_processed'] = full_data['uid'].apply(lambda x: x if x in top_uids else '__RARE__')

    # 按 uid 分组划分训练和验证集（防止 uid 泄漏）
    unique_uids = full_data['uid_processed'].unique()
    np.random.seed(42)
    np.random.shuffle(unique_uids)
    val_ratio = 0.2
    split_idx = int(len(unique_uids) * (1 - val_ratio))
    train_uids = set(unique_uids[:split_idx])
    val_uids = set(unique_uids[split_idx:])

    train_data = full_data[full_data['uid_processed'].isin(train_uids)].copy()
    val_data = full_data[full_data['uid_processed'].isin(val_uids)].copy()

    # 特征设置
    categorical_features = ['site_id', 'gender', 'post_type', 'city_level', 'uid_processed']
    numeric_features = ['statistical_duration', 'publish_weekday', 'age', 'fans_cnt', 'coin_cnt',
                        'video_cnt', 'authority_popularity', 'fans_video_ratio', 'avg_coin_per_video',
                        'avg_fans_per_video', 'site_post', 'site_age_group', 'site_city']

    label_encoders = {}
    categorical_info = {}
    x_categorical_train = {}
    x_categorical_val = {}

    for col in categorical_features:
        le = LabelEncoder()
        all_vals = list(train_data[col].values) + list(val_data[col].values)
        le.fit(all_vals)
        train_data[col + '_enc'] = le.transform(train_data[col])
        val_data[col + '_enc'] = le.transform(val_data[col])
        label_encoders[col] = le
        categorical_info[col + '_enc'] = len(le.classes_)
        x_categorical_train[col + '_enc'] = train_data[col + '_enc'].values
        x_categorical_val[col + '_enc'] = val_data[col + '_enc'].values

    # 数值特征
    scaler = StandardScaler()
    x_numeric_train = scaler.fit_transform(train_data[numeric_features].values)
    x_numeric_val = scaler.transform(val_data[numeric_features].values)

    # 标签
    y_train = np.log(train_data['interaction_cnt'].values + 1)
    y_val = np.log(val_data['interaction_cnt'].values + 1)

    # 保存处理器
    joblib.dump(scaler, params['scaler_save_path'])
    joblib.dump(label_encoders, params['label_encoders_save_path'])

    # 数据加载器
    train_dataset = TabDataset(x_numeric_train, x_categorical_train, y_train, categorical_info)
    val_dataset = TabDataset(x_numeric_val, x_categorical_val, y_val, categorical_info)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)

    # 模型初始化
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = XTransformerWithEmbedding(
        num_numeric_features=len(numeric_features),
        categorical_info=categorical_info,
        embed_dim=16,
        dim=64,
        depth=4,
        heads=4
    ).to(device)

    criterion = LogCoshLoss()
    optimizer = optim.NAdam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)

    # 训练模型
    num_epochs = 50
    best_val_loss = float('inf')
    patience = 5
    wait = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss, total_score = 0.0, 0.0
        epoch_start = time.time()
        for x_cat_batch, x_num_batch, y_batch in train_loader:
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

            total_loss += loss.item()
            total_score += torch.abs(torch.exp(preds) - torch.exp(y_batch)).sum().item()

        model.eval()
        val_loss, val_score = 0.0, 0.0
        with torch.no_grad():
            for x_cat_batch, x_num_batch, y_batch in val_loader:
                for k in x_cat_batch:
                    x_cat_batch[k] = x_cat_batch[k].to(device)
                x_num_batch = x_num_batch.to(device)
                y_batch = y_batch.to(device)

                preds = model(x_num_batch, x_cat_batch)
                loss = criterion(preds, y_batch)
                val_loss += loss.item()
                val_score += torch.abs(torch.exp(preds) - torch.exp(y_batch)).sum().item()

        print(f"Epoch {epoch+1}/{num_epochs} | TrainLoss: {total_loss/len(train_loader):.4f} | "
              f"ValLoss: {val_loss/len(val_loader):.4f} | ValScore: {val_score/len(val_dataset):.4f} | "
              f"Time: {time.time() - epoch_start:.2f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model, params['model_save_path'])
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping triggered.")
                break
