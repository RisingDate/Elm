import numpy as np
import pandas as pd

from Modules.MachineLearning.run.model import CatBoostRegressorModel
from Modules.DeepLearning.run.dataProcess import data_process, CustomDataset

if __name__ == '__main__':
    data_path = 'D:\python_project\Elm\Dataset\B\B.txt'
    # data_path = 'D:\python_project\Elm\Dataset\A\\test_data.txt'
    model_path = 'catboost_model.cbm'
    loaded_model = CatBoostRegressorModel()
    loaded_model.load_model(model_path)
    test_data = data_process(data_path,False)
    categorical_features = ['site_id', 'statistical_duration', 'post_type']
    for col in categorical_features:
        test_data[col] = test_data[col].astype(str)
    # features = ['site_id', 'statistical_duration', 'gender', 'age', 'fans_cnt', 'coin_cnt', 'post_type']  # 替换为实际的特征列名
    # features = ['fans_cnt', 'coin_cnt']  # 替换为实际的特征列名
    features = ['site_id', 'statistical_duration', 'fans_cnt', 'coin_cnt',
                'video_cnt', 'post_type', 'authority_popularity', 'fans_video_ratio',
                'avg_coin_per_video']
    x_test = test_data[features].values
    # 使用加载的模型进行预测
    predictions = loaded_model.predict(x_test)
    prediction_interaction_cnt = predictions

    ids = test_data.iloc[:, 0].values
    combined = np.c_[ids, prediction_interaction_cnt]
    print(combined)

    # 转换为DataFrame并添加表头
    df = pd.DataFrame(combined, columns=["id", "interaction_cnt"])
    # 保存为txt文件（tab分隔）
    # df.to_csv("./results/B/output-250512-with-2-feature.txt", sep='\t', index=False, header=True)
    df.to_csv("output-250512-with-2-feature.csv", index=False, header=True)