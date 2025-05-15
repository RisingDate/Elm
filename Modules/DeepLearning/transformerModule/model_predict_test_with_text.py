import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import joblib
from tqdm import tqdm

from dataProcess import data_process

params = {
    'test_data_path': '../../../Dataset/A/test_data.txt',
    'model_path': '../models/tf-model5.pth',
    'scaler_path': '../models/tf-scaler5.pkl',
    'label_encoders_path': '../models/label_encoders5.pkl'
}

if __name__ == '__main__':
    data_path = params['test_data_path']
    model_path = params['model_path']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.eval()

    test_data = data_process(data_path, True)

    str_features = ['user_site', 'user_post', 'uid']
    label_encoders = joblib.load(params['label_encoders_path'])

    for col in str_features:
        le = label_encoders[col]
        test_data[col] = test_data[col].astype(str).str.strip()
        test_data[col] = test_data[col].apply(lambda x: x if x in le.classes_ else '<UNK>')

        if '<UNK>' not in le.classes_:
            le.classes_ = np.insert(le.classes_, 0, '<UNK>')

        test_data[col] = le.transform(test_data[col])

    # 准备分类输入
    x_cat_dict = {
        col: torch.tensor(test_data[col].values, dtype=torch.long).to(device)
        for col in str_features
    }
    numeric_features = ['site_id', 'statistical_duration', 'publish_weekday', 'gender', 'age', 'fans_cnt', 'coin_cnt',
                        'video_cnt', 'post_type', 'city_level', 'authority_popularity', 'fans_video_ratio',
                        'avg_coin_per_video', 'avg_fans_per_video']
    x_test = test_data[numeric_features].values
    y_test = test_data['interaction_cnt'].values

    scaler = joblib.load(params['scaler_path'])
    x_test_scaled = scaler.transform(x_test)
    x_test_tensor = torch.tensor(x_test_scaled, dtype=torch.float32).to(device)

    x_cat_tensors = [torch.tensor(test_data[col].values, dtype=torch.long) for col in str_features]
    # 构建 DataLoader
    test_dataset = TensorDataset(x_test_tensor, *x_cat_tensors)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

    all_preds = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            x_num = batch[0].to(device)
            x_cat_dict = {
                str_features[i]: batch[i + 1].to(device)
                for i in range(len(str_features))
            }
            log_preds = model(x_num, x_cat_dict).squeeze()
            preds = torch.exp(log_preds) - 1
            all_preds.append(preds.cpu())

    prediction_interaction_cnt = torch.cat(all_preds).floor().int().numpy()
    print(f'len of prediction_interaction_cnt: {len(prediction_interaction_cnt)}')

    absolute_errors = np.abs(prediction_interaction_cnt - y_test)
    print(f'len of absolute_errors: {len(absolute_errors)}')
    absolute_errors_sum = np.sum(absolute_errors)
    score = absolute_errors_sum / len(absolute_errors)
    print(f'score: {score}')
