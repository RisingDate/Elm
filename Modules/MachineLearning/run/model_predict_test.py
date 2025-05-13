import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
import joblib

from Modules.DeepLearning.run.dataProcess import data_process, CustomDataset
from Modules.MachineLearning.run.model import CatBoostRegressorModel

if __name__ == '__main__':
    data_path = 'D:\python_project\Elm\Dataset\A\\test_data.txt'
    model_path = 'catboost_model.cbm'
    loaded_model = CatBoostRegressorModel()
    loaded_model.load_model(model_path)
    test_data = data_process(data_path, False)
    categorical_features = ['site_id', 'statistical_duration', 'post_type']
    for col in categorical_features:
        test_data[col] = test_data[col].astype(str)
    # features = ['site_id', 'statistical_duration', 'gender', 'age', 'fans_cnt', 'coin_cnt', 'post_type']  # 替换为实际的特征列名
    features = ['site_id', 'statistical_duration','fans_cnt', 'coin_cnt',
                'video_cnt', 'post_type', 'authority_popularity', 'fans_video_ratio',
                'avg_coin_per_video',]
    # features = ['fans_cnt', 'coin_cnt']  # 替换为实际的特征列名
    x_test = test_data[features].values
    y_test = test_data['interaction_cnt'].values
    # 使用加载的模型进行预测
    # predictions = loaded_model.predict(x_test)
    # prediction_interaction_cnt = predictions
    # loaded_model.plot_feature_importance(top_n=10)
    print(max(y_test))
    print(f'len of y_test: {len(y_test)}')
    # print(f'len of prediction_interaction_cnt: {len(prediction_interaction_cnt)}')
    zeros_array = np.zeros(len(y_test))
    absolute_errors = np.abs(y_test - zeros_array)
    print(f'len of absolute_errors: {len(absolute_errors)}')
    print(max(absolute_errors))
    print(min(absolute_errors))
    absolute_errors_sum = np.sum(absolute_errors)
    score = absolute_errors_sum / len(absolute_errors)
    print(f'score: {score}')
