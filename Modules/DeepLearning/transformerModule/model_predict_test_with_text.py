import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
import joblib
from collections import Counter

from model import XTransformerWithEmbedding
from dataProcess import data_process

params = {
    'test_data_path': '../../../Dataset/A/test_data.txt',
    'model_path': '../models/tf-model10-with-text.pth',
    'scaler_path': '../models/tf-scaler10-with-text.pkl',
    'label_encoders_path': '../models/label_encoders10.pkl'
}

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载测试数据和预处理器
    test_data = data_process(params['test_data_path'], is_train=True)

    # 添加 uid_processed 特征
    uid_counts = Counter(test_data['uid'])
    test_data['uid_processed'] = test_data['uid'].apply(lambda x: x if uid_counts[x] > 5 else '__RARE__')
    scaler = joblib.load(params['scaler_path'])
    label_encoders = joblib.load(params['label_encoders_path'])

    # 分类特征处理
    categorical_features = ['site_id', 'gender', 'post_type', 'city_level', 'uid_processed']
    x_categorical = {}
    categorical_info = {}

    for col in categorical_features:
        le = label_encoders[col]
        known_classes = set(le.classes_)

        def safe_transform(val):
            return le.transform([val])[0] if val in known_classes else 0  # 安全回退到 index 0

        test_data[col + '_enc'] = test_data[col].apply(safe_transform)
        x_categorical[col + '_enc'] = test_data[col + '_enc'].values

    # 数值特征处理
    numeric_features = ['statistical_duration', 'publish_weekday', 'age', 'fans_cnt', 'coin_cnt',
                        'video_cnt', 'authority_popularity', 'fans_video_ratio', 'avg_coin_per_video',
                        'avg_fans_per_video', 'site_post', 'site_age_group', 'site_city']

    x_numeric = scaler.transform(test_data[numeric_features].values)
    x_numeric_tensor = torch.tensor(x_numeric, dtype=torch.float32).to(device)
    x_categorical_tensor = {
        k: torch.tensor(v, dtype=torch.long).to(device)
        for k, v in x_categorical.items()
    }

    y_true = test_data['interaction_cnt'].values

    # 加载模型并预测
    model = torch.load(params['model_path'], weights_only=False, map_location=device)
    model.eval()

    with torch.no_grad():
        log_preds = model(x_numeric_tensor, x_categorical_tensor).squeeze()
        preds = torch.exp(log_preds) - 1
        preds = preds.floor().int().cpu().numpy()

    abs_err = np.abs(preds - y_true)
    score = abs_err.mean()

    print(f"预测总数: {len(preds)}")
    print(f"平均误差: {score:.4f}")
