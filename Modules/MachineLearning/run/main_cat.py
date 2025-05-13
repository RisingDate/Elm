import numpy as np
import time

import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from Modules.DeepLearning.run.dataProcess import data_process, CustomDataset
from model import SimpleNet, InteractionPredictor, CatBoostRegressorModel

if __name__ == "__main__":
    # 生成示例数据
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split

    # 数据处理
    path = 'D:/python_project/Elm/Dataset/A/train_data.txt'
    train_data = data_process(path)

    features = ['site_id', 'statistical_duration', 'fans_cnt', 'coin_cnt',
                'video_cnt', 'post_type', 'authority_popularity', 'fans_video_ratio',
                'avg_coin_per_video']

    # 处理分类特征
    categorical_features = ['site_id', 'statistical_duration', 'post_type']
    for col in categorical_features:
        train_data[col] = train_data[col].astype(str)

    # 准备特征和目标变量
    X = train_data[features]  # 保留DataFrame格式，让CatBoost自动处理列名
    y = train_data['interaction_cnt'].values
    y = np.log(y + 1)

    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 优化的模型参数
    params = {
        'iterations': 500,
        'learning_rate': 0.1,
        'depth': 6,
        'l2_leaf_reg': 10,
        'subsample': 0.8,
        'colsample_bylevel': 0.8,
        'early_stopping_rounds': 10,
        'verbose': 100,
        'random_seed': 42
    }
    from sklearn.model_selection import ParameterGrid

    # 定义参数搜索空间
    # param_grid = {
    #     'learning_rate': [0.01, 0.05, 0.1],
    #     'depth': [4, 5, 6, 7],
    #     'l2_leaf_reg': [3, 5, 10],
    #     'iterations': [500, 1000]
    # }
    #
    # best_rmse = float('inf')
    # best_params = None
    #
    # # 简单的参数搜索（实际应用中建议使用交叉验证）
    # for params in ParameterGrid(param_grid):
    #     model = CatBoostRegressorModel(
    #         params=params,
    #         categorical_features=categorical_features,
    #         feature_names=features,
    #         random_state=42
    #     )
    #     model.train(X, y)
    #     rmse = model.evaluate(X, y, metric='rmse')
    #     if rmse < best_rmse:
    #         best_rmse = rmse
    #         best_params = params
    #
    # print(f"最优参数: {best_params}")
    # print(f"最优RMSE: {best_rmse:.4f}")
    # 初始化模型
    model = CatBoostRegressorModel(
        params=params,
        categorical_features=categorical_features,
        random_state=42
    )

    # 训练模型（使用验证集）
    model.train(X_train, y_train, eval_set=(X_val, y_val), plot=True)

    # 评估模型
    train_rmse = model.evaluate(X_train, y_train, metric='rmse')
    val_rmse = model.evaluate(X_val, y_val, metric='rmse')
    print(f"训练集RMSE: {train_rmse:.4f}")
    print(f"验证集RMSE: {val_rmse:.4f}")

    # 绘制特征重要性
    model.plot_feature_importance(top_n=20)

    # 保存模型
    model.save_model('catboost_model.cbm')

    # 加载模型
    data_path = 'D:\python_project\Elm\Dataset\A\\test_data.txt'
    model_path = 'catboost_model.cbm'
    loaded_model = CatBoostRegressorModel()
    loaded_model.load_model(model_path)
    test_data = data_process(data_path, False)
    for col in categorical_features:
        test_data[col] = test_data[col].astype(str)
    x_test = test_data[features].values
    y_test = test_data['interaction_cnt'].values
    # 使用加载的模型进行预测
    predictions = loaded_model.predict(x_test)
    prediction_interaction_cnt = predictions
    loaded_model.plot_feature_importance(top_n=10)

    print(f'len of y_test: {len(y_test)}')
    print(f'len of prediction_interaction_cnt: {len(prediction_interaction_cnt)}')

    absolute_errors = np.abs(y_test - prediction_interaction_cnt)
    print(f'len of absolute_errors: {len(absolute_errors)}')
    absolute_errors_sum = np.sum(absolute_errors)
    score = absolute_errors_sum / len(absolute_errors)
    print(f'score: {score}')

    prediction_interaction_cnt = predictions

    ids = test_data.iloc[:, 0].values
    combined = np.c_[ids, prediction_interaction_cnt]
    print(combined)

    # 转换为DataFrame并添加表头
    df = pd.DataFrame(combined, columns=["id", "interaction_cnt"])
    # 保存为txt文件（tab分隔）
    # df.to_csv("./results/B/output-250512-with-2-feature.txt", sep='\t', index=False, header=True)
    df.to_csv("output-250512-with-2-feature.csv", index=False, header=True)
