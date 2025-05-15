import time

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
from catboost import CatBoostRegressor
import torch
import joblib
from model import XTransformer
from dataProcess import CustomDataset, data_process
from torch.utils.data import DataLoader
import lightgbm as lgb

# 参数
data_path = '../../../Dataset/A/train_data.txt'
transformer_model_path = '../models/xtransform_fold.pth'  # 替换为你保存的模型
catboost_model_path = '../models/catboost_model.cbm'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 特征
categorical_features = ['site_id', 'gender', 'post_type', 'city_level', 'user_site', 'user_post']
numeric_features = ['statistical_duration', 'publish_weekday', 'age', 'fans_cnt', 'coin_cnt',
                    'video_cnt', 'authority_popularity', 'fans_video_ratio', 'avg_coin_per_video',
                    'avg_fans_per_video', 'site_post', 'site_age_group', 'site_city']
all_features = categorical_features + numeric_features

# all_features = ['site_id', 'statistical_duration', 'publish_weekday', 'gender', 'age', 'fans_cnt', 'coin_cnt',
#                 'video_cnt', 'post_type', 'city_level', 'authority_popularity', 'fans_video_ratio',
#                 'avg_coin_per_video', 'avg_fans_per_video', 'site_post', 'site_age_group', 'site_city']

# 加载数据
df = data_process(data_path, is_train=True)
df['interaction_cnt_log'] = np.log1p(df['interaction_cnt'])

# 编码分类特征
label_encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# 保存编码器
joblib.dump(label_encoders, '../models/stacking_labelenc.pkl')

# 标准化数值特征
scaler = StandardScaler()
df[numeric_features] = scaler.fit_transform(df[numeric_features])
joblib.dump(scaler, '../models/stacking_scaler.pkl')

X = df[all_features]
y = df['interaction_cnt_log'].values

# 准备融合输入
oof_cb = np.zeros(len(df))
oof_tf = np.zeros(len(df))

print('============正在训练融合模型=============')
kf = KFold(n_splits=5, shuffle=True, random_state=42)
epochs = 50

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"Fold: {fold+1}")

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    print('============CatBoost训练中=============')
    # ✅ 1. CatBoost
    cb_model = CatBoostRegressor(verbose=0)
    cb_model.fit(X_train, y_train, cat_features=categorical_features)
    oof_cb[val_idx] = cb_model.predict(X_val)
    catboost_model_path = f'../models/catboost_fold{fold}.cbm'
    cb_model.save_model(catboost_model_path)

    print('============XTransformer训练中=============')
    # ✅ 2. XTransformer
    xtrain_tf = torch.tensor(X_train.values, dtype=torch.float32).to(device)
    ytrain_tf = torch.tensor(y_train, dtype=torch.float32).to(device)
    xval_tf = torch.tensor(X_val.values, dtype=torch.float32).to(device)

    model = XTransformer(input_dim=X.shape[1], dim=64, depth=4, heads=4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = torch.nn.HuberLoss()
    loader = DataLoader(CustomDataset(xtrain_tf, ytrain_tf), batch_size=64, shuffle=True)

    model.train()
    for epoch in range(epochs):  # 可增加轮数
        epoch_start_time = time.time()
        running_loss = 0
        model_preds = []
        y_true = []

        for xb, yb in loader:
            yb = yb.squeeze()
            optimizer.zero_grad()
            pred = model(xb).squeeze()
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            model_preds.extend(pred.detach().cpu().numpy())
            y_true.extend(yb.detach().cpu().numpy())

        epoch_end_time = time.time()
        mae = mean_absolute_error(np.expm1(y_true), np.expm1(model_preds))

        print(f'Fold: {fold + 1} -> Epoch {epoch+1}/{epochs} | '
              f'Loss: {running_loss / len(loader)} | '
              f'MAE Score: {mae:.4f} | '
              f'Running Time: {epoch_end_time - epoch_start_time}s')

    model.eval()
    with torch.no_grad():
        pred_val_tf = model(xval_tf).squeeze().cpu().numpy()
        oof_tf[val_idx] = pred_val_tf
    transformer_save_path = f'../models/xtransform_fold{fold}.pth'
    torch.save(model.state_dict(), transformer_save_path)

# 构建 stacking 输入
stacking_X = np.vstack([oof_cb, oof_tf]).T  # shape: (n_samples, 2)
stacking_y = np.expm1(y)  # 回到原始空间

print('============正在训练LightGBM=============')
# ✅ 使用 LightGBM 做融合模型
meta_model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.1)
meta_model.fit(stacking_X, stacking_y)

# 评估融合结果
final_preds = meta_model.predict(stacking_X)
mae = mean_absolute_error(stacking_y, final_preds)
print(f"\n融合模型 MAE：{mae:.4f}")

# 保存融合模型
joblib.dump(meta_model, '../models/stacking_meta_lgb.pkl')
