import pandas as pd
import numpy as np
import joblib
import torch
from catboost import CatBoostRegressor
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from model import XTransformer

from dataProcess import data_process

# ===== 路径配置 =====
data_path = '../../../Dataset/A/train_data.txt'
catboost_model_path = '../models/stacking-catboost_model.cbm'
transformer_model_path = '../models/tf-model7.pth'

scaler_path = '../models/tf-scaler7.pkl'
stacking_model_path = '../models/stacking_meta_lgb.pkl'

# ===== 数值特征（共17个）=====
features = [
    'site_id', 'statistical_duration', 'publish_weekday', 'gender', 'age',
    'fans_cnt', 'coin_cnt', 'video_cnt', 'post_type', 'city_level',
    'authority_popularity', 'fans_video_ratio', 'avg_coin_per_video',
    'avg_fans_per_video', 'site_post', 'site_age_group', 'site_city'
]

# ===== 加载并预处理训练数据 =====
df = data_process(data_path, is_train=True)
df['interaction_cnt_log'] = np.log1p(df['interaction_cnt'])

# 加载 scaler 并应用
scaler = joblib.load(scaler_path)
df[features] = scaler.transform(df[features])

X = df[features]
y = df['interaction_cnt_log'].values  # 注意：stacking_y 是原始空间

# ===== 1️⃣ CatBoost 推理 =====
cb_model = CatBoostRegressor()
cb_model.load_model(catboost_model_path)
oof_cb = cb_model.predict(X)

# ===== 2️⃣ XTransformer 推理 =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x_tensor = torch.tensor(X.values, dtype=torch.float32).to(device)
model = torch.load(transformer_model_path, map_location=device, weights_only=False)
model.eval()
with torch.no_grad():
    oof_tf = model(x_tensor).squeeze().cpu().numpy()

# ===== 3️⃣ 构建 stacking 特征 + 训练融合器 =====
stacking_X = np.vstack([oof_cb, oof_tf]).T
meta_model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05)
meta_model.fit(stacking_X, y)

# ===== 4️⃣ 保存融合器 + 输出 MAE =====
joblib.dump(meta_model, stacking_model_path)
print(f"✅ 融合模型已保存至: {stacking_model_path}")

# 还原预测和真实标签
preds = meta_model.predict(stacking_X)
final_preds = np.expm1(preds)  # 对预测结果应用 expm1()
final_preds = np.clip(final_preds, 0, 1e8).astype(int)
original_y = np.expm1(y)  # 对真实标签应用 expm1()



# 计算 MAE
mae = mean_absolute_error(original_y, final_preds)
print(f"📊 融合器训练集 MAE: {mae:.4f}")
