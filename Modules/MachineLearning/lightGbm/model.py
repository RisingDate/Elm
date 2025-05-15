import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

from dataProcess import data_process

# 配置路径
train_data_path = '../../../Dataset/A/train_data.txt'
scaler_path = '../models/lgb_scaler.pkl'
label_enc_path = '../models/lgb_labelenc.pkl'
model_path = '../models/lightgbm_model.txt'

# 加载数据
df = data_process(train_data_path, is_train=True)

# 特征列
categorical_features = ['site_id', 'gender', 'post_type', 'city_level', 'user_site', 'user_post']
numeric_features = ['statistical_duration', 'publish_weekday', 'age', 'fans_cnt', 'coin_cnt',
                    'video_cnt', 'authority_popularity', 'fans_video_ratio', 'avg_coin_per_video',
                    'avg_fans_per_video', 'site_post', 'site_age_group', 'site_city']
all_features = categorical_features + numeric_features

# 编码分类特征
label_encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le
joblib.dump(label_encoders, label_enc_path)

# 标准化数值特征
scaler = StandardScaler()
df[numeric_features] = scaler.fit_transform(df[numeric_features])
joblib.dump(scaler, scaler_path)

# 构造训练集
X = df[all_features]
y = np.log1p(df['interaction_cnt'])  # log变换目标值

train_dataset = lgb.Dataset(X, label=y, categorical_feature=categorical_features)

# 模型参数
params = {
    'objective': 'regression',
    'metric': 'mae',
    'learning_rate': 0.05,
    'max_depth': 6,
    'num_leaves': 31,
    'verbose': -1
}

# 训练模型
print("正在训练 LightGBM 模型...")
model = lgb.train(params, train_dataset, num_boost_round=1000)
model.save_model(model_path)
print(f"模型已保存至: {model_path}")
