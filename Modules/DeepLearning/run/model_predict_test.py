import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
import joblib

from dataProcess import data_process, CustomDataset


if __name__ == '__main__':
    data_path = '../../../Dataset/A/test_data.txt'
    model_path = '../models/model6.pth'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.eval()

    test_data = data_process(data_path, False)
    # features = ['site_id', 'statistical_duration', 'gender', 'age', 'fans_cnt', 'coin_cnt', 'post_type']  # 替换为实际的特征列名
    features = ['site_id', 'statistical_duration', 'publish_weekday', 'gender', 'age', 'fans_cnt', 'coin_cnt', 'post_type']  # 替换为实际的特征列名
    # features = ['fans_cnt', 'coin_cnt']  # 替换为实际的特征列名
    x_test = test_data[features].values
    y_test = test_data['interaction_cnt'].values

    # scaler = StandardScaler()
    scaler = joblib.load('../models/scaler6.pkl')
    x_test_scaled = scaler.transform(x_test)
    x_test_tensor = torch.tensor(x_test_scaled, dtype=torch.float32).to(device)

    with torch.no_grad():
        log_predictions = model(x_test_tensor)
        predictions = torch.exp(log_predictions) - 1
        predictions = predictions.floor().int()

    prediction_interaction_cnt = predictions.numpy().flatten()
    print(f'len of y_test: {len(y_test)}')
    print(f'len of prediction_interaction_cnt: {len(prediction_interaction_cnt)}')

    absolute_errors = np.abs(y_test - prediction_interaction_cnt)
    print(f'len of absolute_errors: {len(absolute_errors)}')
    absolute_errors_sum = np.sum(absolute_errors)
    score = absolute_errors_sum / len(absolute_errors)
    print(f'score: {score}')
