import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool

from dataProcess import data_process

# 加载数据
data_path = '../../../Dataset/A/train_data.txt'
df = data_process(data_path, is_train=True)

# 对目标变量做 log1p 变换（提升鲁棒性）
df['interaction_cnt_log'] = np.log1p(df['interaction_cnt'])

# 特征列定义
categorical_features = ['site_id', 'gender', 'post_type', 'city_level', 'user_site', 'user_post']
numeric_features = ['statistical_duration', 'publish_weekday', 'age', 'fans_cnt', 'coin_cnt',
                    'video_cnt', 'authority_popularity', 'fans_video_ratio', 'avg_coin_per_video',
                    'avg_fans_per_video', 'site_post', 'site_age_group', 'site_city']

all_features = categorical_features + numeric_features

all_features = ['site_id', 'statistical_duration', 'publish_weekday', 'gender', 'age', 'fans_cnt', 'coin_cnt',
                'video_cnt', 'post_type', 'city_level', 'authority_popularity', 'fans_video_ratio', 'avg_coin_per_video',
                'avg_fans_per_video', 'site_post', 'site_age_group', 'site_city']

# 构造 CatBoost Pool 数据结构
train_pool = Pool(df[all_features], label=df['interaction_cnt_log'])

# 初始化并训练 CatBoost 模型
model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    loss_function='RMSE',
    verbose=100
)

model.fit(train_pool)

# 保存模型
model.save_model('../models/stacking-catboost_model.cbm')
print("✅ 模型已训练并保存到: ../models/stacking-catboost_model.cbm")
