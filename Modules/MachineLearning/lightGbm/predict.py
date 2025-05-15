import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from sklearn.metrics import mean_absolute_error

from dataProcess import data_process

# 配置路径
# test_data_path = '../../../Dataset/A/test_data.txt'
test_data_path = '../../../Dataset/B/B.txt'
scaler_path = '../models/lgb_scaler.pkl'
label_enc_path = '../models/lgb_labelenc.pkl'
model_path = '../models/lightgbm_model.txt'
save_txt_path = '../results/lightgbm_output.txt'
save_csv_path = '../results/lightgbm_output.csv'

# 加载数据
df = data_process(test_data_path, is_train=False)

# 特征列
categorical_features = ['site_id', 'gender', 'post_type', 'city_level', 'user_site', 'user_post']
numeric_features = ['statistical_duration', 'publish_weekday', 'age', 'fans_cnt', 'coin_cnt',
                    'video_cnt', 'authority_popularity', 'fans_video_ratio', 'avg_coin_per_video',
                    'avg_fans_per_video', 'site_post', 'site_age_group', 'site_city']
all_features = categorical_features + numeric_features

# 加载编码器
label_encoders = joblib.load(label_enc_path)
for col in categorical_features:
    le = label_encoders[col]
    known_classes = set(le.classes_)
    df[col] = df[col].apply(lambda x: le.transform([x])[0] if x in known_classes else 0)

# 加载 scaler 并归一化数值特征
scaler = joblib.load(scaler_path)
df[numeric_features] = scaler.transform(df[numeric_features])

# 加载模型
model = lgb.Booster(model_file=model_path)

# 预测
X = df[all_features]
log_preds = model.predict(X)
preds = np.expm1(log_preds).clip(0, 1e8).astype(int)
ids = df.iloc[:, 0].values  # 默认第一列为 ID
results = pd.DataFrame({'id': ids, 'interaction_cnt': preds})

if 'interaction_cnt' in df.columns and test_data_path != '../../../Dataset/B/B.txt':
    abs_err = np.abs(preds - df['interaction_cnt'].values)
    mae = abs_err.mean()
    print(f"lightGbm MAE Score: {mae:.4f}")
else:
    # 保存结果
    results.to_csv(save_txt_path, sep='\t', index=False, header=True)
    results.to_csv(save_csv_path, index=False, header=True)
    print(f"lightGbm 预测结果已保存至：\n- {save_txt_path}\n- {save_csv_path}")
