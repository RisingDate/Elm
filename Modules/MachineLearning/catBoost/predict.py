import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error

from dataProcess import data_process

# 参数路径
params = {
    'test_data_path': '../../../Dataset/A/test_data.txt',
    # 'test_data_path': '../../../Dataset/B/B.txt',
    'model_path': '../models/catboost_model.cbm',
    'save_txt_path': '../results/catboost_output.txt',
    'save_csv_path': '../results/catboost_output.csv'
}

# 特征设置（需与训练时一致）
categorical_features = ['site_id', 'gender', 'post_type', 'city_level', 'user_site', 'user_post']
numeric_features = ['statistical_duration', 'publish_weekday', 'age', 'fans_cnt', 'coin_cnt',
                    'video_cnt', 'authority_popularity', 'fans_video_ratio', 'avg_coin_per_video',
                    'avg_fans_per_video', 'site_post', 'site_age_group', 'site_city']
all_features = categorical_features + numeric_features

# 读取测试数据
if params['test_data_path'] != '../../../Dataset/B/B.txt':
    df = data_process(params['test_data_path'], is_train=True)
else:
    df = data_process(params['test_data_path'], is_train=False)
print(f"测试样本数：{len(df)}")

# 加载训练好的 CatBoost 模型
model = CatBoostRegressor()
model.load_model(params['model_path'])
print("✅ 模型已加载")

# 模型预测
log_preds = model.predict(df[all_features])
preds = np.expm1(log_preds)  # 反log变换
preds = np.clip(preds, 0, 1e8).astype(int)  # 限制范围 + 转为整数

# 拼接 ID + 预测结果
ids = df.iloc[:, 0].values  # 默认第一列为 ID
results = pd.DataFrame({'id': ids, 'interaction_cnt': preds})

if 'interaction_cnt' in df.columns and params['test_data_path'] != '../../../Dataset/B/B.txt':
    abs_err = np.abs(preds - df['interaction_cnt'].values)
    mae = abs_err.mean()
    print(f"catBoost MAE Score: {mae:.4f}")
else:
    # 保存结果
    results.to_csv(params['save_txt_path'], sep='\t', index=False, header=True)
    results.to_csv(params['save_csv_path'], index=False, header=True)
    print(f"catBoost 预测结果已保存至：\n- {params['save_txt_path']}\n- {params['save_csv_path']}")
