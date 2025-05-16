import pandas as pd
import numpy as np
import joblib
import torch
from catboost import CatBoostRegressor
import lightgbm as lgb
from model import XTransformer
from dataProcess import data_process
from sklearn.metrics import mean_absolute_error

# ===== 路径配置 =====
test_data_path = '../../../Dataset/B/B.txt'
catboost_model_path = '../models/stacking-catboost_model.cbm'
transformer_model_path = '../models/tf-model-all_data.pth'
stacking_model_path = '../models/stacking_meta_lgb.pkl'
scaler_path = '../models/tf-scaler-all_data.pkl'


save_path = '../results/B/stacking_predict_result.txt'

# ===== 数值特征（共17个）=====
features = [
    'site_id', 'statistical_duration', 'publish_weekday', 'gender', 'age',
    'fans_cnt', 'coin_cnt', 'video_cnt', 'post_type', 'city_level',
    'authority_popularity', 'fans_video_ratio', 'avg_coin_per_video',
    'avg_fans_per_video', 'site_post', 'site_age_group', 'site_city'
]

# ===== 加载并预处理测试数据 =====
df = data_process(test_data_path, is_train=False)
scaler = joblib.load(scaler_path)
df[features] = scaler.transform(df[features])
X = df[features]

# ===== 1️⃣ CatBoost 推理 =====
cb_model = CatBoostRegressor()
cb_model.load_model(catboost_model_path)
pred_cb = cb_model.predict(X)

# ===== 2️⃣ XTransformer 推理 =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x_tensor = torch.tensor(X.values, dtype=torch.float32).to(device)
transformer_model = torch.load(transformer_model_path, map_location=device, weights_only=False)
transformer_model.eval()
with torch.no_grad():
    pred_tf = transformer_model(x_tensor).squeeze().cpu().numpy()

# ===== 3️⃣ 构建 stacking 特征 + 使用融合器预测 =====
stacking_X = np.vstack([pred_cb, pred_tf]).T
meta_model = joblib.load(stacking_model_path)
stacking_preds_log = meta_model.predict(stacking_X)
stacking_preds = np.expm1(stacking_preds_log)
stacking_preds = np.clip(stacking_preds, 0, 1e8).astype(int)

# ===== 4️⃣ 保存预测结果 =====
output_df = pd.DataFrame({
    'id': df.iloc[:, 0],  # 假设第一列是 id
    'interaction_cnt': stacking_preds
})
# output_df.to_csv(save_path, index=False)
output_df.to_csv(save_path, sep='\t', index=False, header=True)
output_df.to_csv('../results/B/stacking_predict_result.csv', index=False, header=True)
print(f"✅ 预测结果已保存至: {save_path}")

# ===== 5️⃣ 如有真实标签，计算 MAE =====
# if 'interaction_cnt' in df.columns:
#     true_vals = np.expm1(np.log1p(df['interaction_cnt'].values))  # 防止误差
#     mae = mean_absolute_error(true_vals, stacking_preds)
#     print(f"📊 测试集 MAE: {mae:.4f}")
# else:
#     print("⚠️ 测试集中无 interaction_cnt 列，跳过 MAE 计算。")
