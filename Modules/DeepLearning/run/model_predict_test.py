import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
import joblib

from dataProcess import data_process, CustomDataset


if __name__ == '__main__':
    data_path = '../../../Dataset/A/test_data.txt'
    model_path = '../models/model10.pth'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.eval()

    test_data = data_process(data_path, True)
    # features = ['site_id', 'statistical_duration', 'publish_weekday', 'gender', 'age', 'fans_cnt', 'coin_cnt', 'post_type']
    # features = ['site_id', 'statistical_duration', 'publish_weekday', 'gender', 'age', 'fans_cnt', 'coin_cnt',
    #             'video_cnt', 'post_type', 'city_level', 'authority_popularity', 'fans_video_ratio',
    #             'avg_coin_per_video', 'avg_fans_per_video']
    features = ['site_id', 'statistical_duration', 'fans_cnt', 'coin_cnt', 'video_cnt', 'post_type',
                'authority_popularity', 'fans_video_ratio', 'avg_coin_per_video', 'avg_fans_per_video']
    x_test = test_data[features].values
    y_test = test_data['interaction_cnt'].values

    # scaler = StandardScaler()
    scaler = joblib.load('../models/scaler10.pkl')
    x_test_scaled = scaler.transform(x_test)
    x_test_tensor = torch.tensor(x_test_scaled, dtype=torch.float32).to(device)

    with torch.no_grad():
        log_predictions = model(x_test_tensor)
        predictions = torch.exp(log_predictions) - 1
        predictions = predictions.floor().int()

    zeros = np.zeros(y_test.shape)
    prediction_interaction_cnt = predictions.numpy().flatten()
    print(f'len of prediction_interaction_cnt: {len(prediction_interaction_cnt)}')

    # zeros = np.zeros(prediction_interaction_cnt.shape)
    absolute_errors = np.abs(zeros - y_test)
    print(f'len of absolute_errors: {len(absolute_errors)}')
    absolute_errors_sum = np.sum(absolute_errors)
    score = absolute_errors_sum / len(absolute_errors)
    print(f'score: {score}')
