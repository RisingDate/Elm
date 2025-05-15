import pandas as pd
import numpy as np
import joblib
import torch
from catboost import CatBoostRegressor
import lightgbm as lgb
from model import XTransformer
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder, StandardScaler

from dataProcess import data_process

# ===== 参数配置 =====
test_path = '../../../Dataset/A/test_data.txt'
catboost_model_path = '../models/catboost_model.cbm'
transformer_model_path = '../models/xtransform_fold0.pth'  # 替换为你保存的模型
stacking_model_path = '../models/stacking_meta_lgb.pkl'
label_enc_path = '../models/stacking_labelenc.pkl'
scaler_path = '../models/stacking_scaler.pkl'
save_csv_path = '../results/B/stacking_predict_result.csv'

# ===== 特征配置 =====
categorical_features = ['site_id', 'gender', 'post_type', 'city_level', 'user_site', 'user_post']
numeric_features = ['statistical_duration', 'publish_weekday', 'age', 'fans_cnt', 'coin_cnt',
                    'video_cnt', 'authority_popularity', 'fans_video_ratio', 'avg_coin_per_video',
                    'avg_fans_per_video', 'site_post', 'site_age_group', 'site_city']
all_features = categorical_features + numeric_features

# ===== 加载测试数据 =====
df = data_process(test_path)
X_orig = df.copy()

# ===== 预处理：编码分类变量 =====
label_encoders = joblib.load(label_enc_path)
for col in categorical_features:
    le = label_encoders[col]
    df[col] = df[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else 0)

# ===== 标准化数值变量 =====
scaler = joblib.load(scaler_path)
df[numeric_features] = scaler.transform(df[numeric_features])

# ===== 一级模型预测：CatBoost =====
cb_model = CatBoostRegressor()
cb_model.load_model(catboost_model_path)
catboost_preds = cb_model.predict(df[all_features])

# ===== 一级模型预测：Transformer =====
x_numeric_tensor = torch.tensor(df[all_features].values, dtype=torch.float32)
transformer_model = XTransformer(input_dim=len(all_features), dim=64, depth=4, heads=4)
transformer_model.load_state_dict(torch.load(transformer_model_path, map_location='cpu'))
transformer_model.eval()
with torch.no_grad():
    transformer_preds = transformer_model(x_numeric_tensor).squeeze().numpy()

# ===== 构建 Stacking 特征 =====
stacking_input = np.vstack([catboost_preds, transformer_preds]).T  # shape: (n_samples, 2)

# ===== 二级融合模型预测 =====
meta_model = joblib.load(stacking_model_path)
final_preds = meta_model.predict(stacking_input)
final_preds = np.clip(final_preds, 0, 1e8).astype(int)

# ===== MAE 评分（如有真实标签）=====
if 'interaction_cnt' in df.columns:
    mae = mean_absolute_error(df['interaction_cnt'].values, final_preds)
    print(f"📊 最终融合模型 MAE: {mae:.4f}")
else:
    print("⚠️ 测试集中无真实标签，跳过 MAE 评估")

# ===== 输出保存 =====
output_df = pd.DataFrame({
    'id': X_orig.iloc[:, 0],  # 假设第一列是 id
    'interaction_cnt': final_preds
})
output_df.to_csv(save_csv_path, index=False)
print(f"✅ 预测已保存至: {save_csv_path}")
