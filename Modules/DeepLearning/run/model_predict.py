import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from dataProcess import data_process, CustomDataset


if __name__ == '__main__':
    data_path = '../../../Dataset/B/B.txt'
    model_path = '../models/model9.pth'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.eval()

    test_data = data_process(data_path, is_train=False)
    print('Length of test_data', len(test_data))
    # features = ['site_id', 'statistical_duration', 'gender', 'age', 'fans_cnt', 'coin_cnt', 'post_type']  # 替换为实际的特征列名
    # features = ['fans_cnt', 'coin_cnt']  # 替换为实际的特征列名
    features = ['site_id', 'statistical_duration', 'publish_weekday', 'gender', 'age', 'fans_cnt', 'coin_cnt',
                'video_cnt', 'post_type', 'city_level', 'authority_popularity', 'fans_video_ratio',
                'avg_coin_per_video', 'avg_fans_per_video']
    # features = ['site_id', 'statistical_duration', 'fans_cnt', 'coin_cnt', 'video_cnt', 'post_type',
    #             'authority_popularity', 'fans_video_ratio', 'avg_coin_per_video', 'avg_fans_per_video']
    x_test = test_data[features].values

    scaler = joblib.load('../models/scaler9.pkl')
    x_test_scaled = scaler.transform(x_test)
    x_test_tensor = torch.tensor(x_test_scaled, dtype=torch.float32).to(device)

    with torch.no_grad():
        log_predictions = model(x_test_tensor)
        predictions = torch.exp(log_predictions) - 1

        mask = (predictions < 0) | (predictions > 1e8)
        predictions[mask] = 0
        predictions = predictions.floor().int()

    prediction_interaction_cnt = predictions.numpy().flatten()
    prediction_interaction_cnt = np.zeros(prediction_interaction_cnt.shape)

    ids = test_data.iloc[:, 0].values
    combined = np.c_[ids, prediction_interaction_cnt]
    print(combined)

    # 转换为DataFrame并添加表头
    df = pd.DataFrame(combined, columns=["id", "interaction_cnt"])
    # 保存为txt文件（tab分隔）
    df.to_csv("../results/B/output-250514-1-zeros.txt", sep='\t', index=False, header=True)
    df.to_csv("../results/B/output-250514-1-zeros.csv", index=False, header=True)