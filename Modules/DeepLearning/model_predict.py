import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from dataProcess import data_process, CustomDataset


if __name__ == '__main__':
    data_path = '../../Dataset/A/A.txt'
    model_path = './models/model1.pth'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.eval()

    test_data = data_process(data_path)
    features = ['site_id', 'statistical_duration', 'gender', 'age', 'fans_cnt', 'coin_cnt', 'post_type']  # 替换为实际的特征列名
    x_test = test_data[features]

    scaler = StandardScaler()
    x_test_scaled = scaler.fit_transform(x_test.values)
    x_test_tensor = torch.tensor(x_test_scaled, dtype=torch.float32).to(device)

    with torch.no_grad():
        log_predictions = model(x_test_tensor)
        predictions = torch.exp(log_predictions) - 1

    prediction_interaction_cnt = predictions.numpy().flatten()

    ids = test_data.iloc[:, 0].values
    combined = np.c_[ids, prediction_interaction_cnt]
    print(combined)

    # 转换为DataFrame并添加表头
    df = pd.DataFrame(combined, columns=["id", "interaction_cnt"])
    # 保存为txt文件（tab分隔）
    df.to_csv("./results/output-model1.txt", sep="\t", index=False, header=True)