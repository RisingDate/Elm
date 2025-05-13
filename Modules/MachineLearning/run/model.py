import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, r2_score
import torch.nn.functional as F
from catboost import CatBoostRegressor, Pool


# 定义全连接神经网络模型
class SimpleNet(nn.Module):
    def __init__(self, input_size):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out


class InteractionPredictor(nn.Module):
    def __init__(self, input_size):
        super(InteractionPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.drop1 = nn.Dropout(0.4)

        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.drop2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)

        self.output = nn.Linear(64, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.drop1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.drop2(x)
        x = F.relu(self.bn3(self.fc3(x)))
        return self.output(x)


class DynamicBatchNorm(nn.BatchNorm1d):
    def __init__(self, num_features):
        super().__init__(num_features)
        self.last_batch_size = None

    def forward(self, x):
        if self.training and (x.shape[0] != self.last_batch_size):
            # 动态调整running stats
            self.reset_running_stats()
            self.last_batch_size = x.shape[0]
        return super().forward(x)


class EnhancedInteractionPredictor(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_proj = nn.Linear(input_size, 256)

        self.block1 = nn.Sequential(
            DynamicBatchNorm(256),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128)
        )

        self.block2 = nn.Sequential(
            DynamicBatchNorm(128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64)
        )

        self.output = nn.Sequential(
            DynamicBatchNorm(64),
            nn.Linear(64, 1),
            nn.Softplus()
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)  # 更适合回归任务的初始化
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.block1(x) + x[:, :128]  # 残差连接
        x = self.block2(x) + x[:, :64]
        return self.output(x)


from catboost import CatBoostRegressor

import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class CatBoostRegressorModel:
    """
    CatBoost回归模型封装类，用于训练、评估和预测
    """

    def __init__(self, params=None, categorical_features=None, feature_names = None, random_state=42):
        """
        初始化模型

        参数:
            params: 模型参数，默认为None
            categorical_features: 分类特征列名列表，默认为None
            random_state: 随机种子
        """
        # 设置默认参数
        if params is None:
            self.params = {
                'iterations': 1000,
                'learning_rate': 0.05,
                'depth': 6,
                'l2_leaf_reg': 3,
                'early_stopping_rounds': 20,
                'verbose': 100,
                'random_seed': random_state,
            }
        else:
            self.params = params

        self.categorical_features = categorical_features
        self.random_state = random_state
        self.model = None
        self.feature_importance = None
        self.feature_names = feature_names  # 初始化特征名称属性

    def train(self, X, y, eval_set=None, plot=False):
        """
        训练模型

        参数:
            X: 特征数据
            y: 目标变量
            eval_set: 评估数据集，格式为(X_val, y_val)，默认为None
            plot: 是否绘制训练过程，默认为False
        """
        # 划分训练集和验证集
        if eval_set is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=self.random_state
            )
            eval_set = (X_val, y_val)
        else:
            X_train, y_train = X, y

        # 初始化模型
        self.model = CatBoostRegressor(**self.params)

        # 训练模型
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            cat_features=self.categorical_features,
            use_best_model=True
        )

        # 保存特征重要性
        self.feature_importance = self.model.get_feature_importance()

        # 绘制训练过程
        if plot:
            self.plot_training_history()

        return self

    def predict(self, X):
        """
        预测

        参数:
            X: 特征数据，可以是DataFrame、numpy数组或Pool对象

        返回:
            预测结果
        """
        if self.model is None:
            raise Exception("模型尚未训练，请先调用train方法")

        # 验证输入数据
        if isinstance(X, pd.DataFrame):
            # 检查DataFrame的列是否与训练时一致
            if self.feature_names is not None:
                missing_features = set(self.feature_names) - set(X.columns)
                if missing_features:
                    print(f"警告: 输入数据缺少以下特征: {missing_features}")

                # 确保列顺序一致
                X = X[self.feature_names]

            # 确保分类特征类型正确
            if self.categorical_features:
                for col in self.categorical_features:
                    if col in X.columns and X[col].dtype != 'object':
                        print(f"警告: 列 {col} 类型已转换为object")
                        X[col] = X[col].astype(str)

        # elif isinstance(X, np.ndarray):
        #     # 检查数组维度
        #     if X.ndim == 1:
        #         X = X.reshape(1, -1)  # 单样本预测
        #
        #     if X.shape[1] != len(self.feature_names):
        #         raise ValueError(f"输入特征数量({X.shape[1]})与训练时({len(self.feature_names)})不一致")

        # 打印输入数据的基本信息
        print(f"预测输入形状: {X.shape}")
        if isinstance(X, pd.DataFrame):
            print(f"输入特征: {list(X.columns)}")

        # 执行预测
        predictions = self.model.predict(X)

        # 检查预测结果
        if len(predictions) == 0:
            print("警告: 预测结果为空列表")
            print(f"输入数据类型: {type(X)}, 形状: {X.shape}")
        predictions = [int(x) for x in predictions]
        return predictions

    def evaluate(self, X, y, metric='rmse'):
        """
        评估模型性能

        参数:
            X: 特征数据
            y: 真实标签
            metric: 评估指标，可选 'rmse', 'mae', 'r2'

        返回:
            评估结果
        """
        y_pred = self.predict(X)

        if metric.lower() == 'rmse':
            return np.sqrt(mean_squared_error(y, y_pred))
        elif metric.lower() == 'mae':
            return mean_absolute_error(y, y_pred)
        elif metric.lower() == 'r2':
            return r2_score(y, y_pred)
        else:
            raise ValueError("不支持的评估指标，请选择 'rmse', 'mae' 或 'r2'")

    def plot_feature_importance(self, top_n=20):
        """
        绘制特征重要性图

        参数:
            top_n: 显示前n个重要特征，默认为20
        """
        if self.feature_importance is None:
            raise Exception("模型尚未训练，请先调用train方法")

        feature_names = self.model.feature_names_
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': self.feature_importance
        })
        importance_df = importance_df.sort_values('Importance', ascending=False).head(top_n)

        plt.figure(figsize=(10, 6))
        plt.barh(importance_df['Feature'], importance_df['Importance'])
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title('Feature Importance')
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    def plot_training_history(self):
        """
        绘制训练历史
        """
        if self.model is None:
            raise Exception("模型尚未训练，请先调用train方法")

        eval_metrics = self.model.get_evals_result()
        iterations = list(range(1, len(eval_metrics['validation']['RMSE']) + 1))

        plt.figure(figsize=(10, 6))
        plt.plot(iterations, eval_metrics['validation']['RMSE'], label='Validation RMSE')
        plt.xlabel('Iteration')
        plt.ylabel('RMSE')
        plt.title('Training History')
        plt.legend()
        plt.grid(linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    def get_best_iteration(self):
        """
        获取最佳迭代次数
        """
        if self.model is None:
            raise Exception("模型尚未训练，请先调用train方法")

        return self.model.get_best_iteration()

    def save_model(self, path):
        """
        保存模型

        参数:
            path: 模型保存路径
        """
        if self.model is None:
            raise Exception("模型尚未训练，请先调用train方法")

        self.model.save_model(path)

    def load_model(self, path):
        """
        加载模型

        参数:
            path: 模型加载路径
        """
        self.model = CatBoostRegressor()
        self.model.load_model(path)
        self.feature_importance = self.model.get_feature_importance()
        return self


# 使用示例


# # 评估模型
# model.eval()
# y_pred = []
# with torch.no_grad():
#     for inputs, _ in test_loader:
#         inputs = inputs.to(device)
#         outputs = model(inputs)
#         y_pred.extend(outputs.cpu().numpy().flatten())
#
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)
# print(f'Mean Squared Error: {mse}')
# print(f'R-squared: {r2}')
#
#
#
# # 保存整个模型
# torch.save(model, './models/model1.pth')
# # 加载整个模型
# loaded_model = torch.load('full_model.pth')
# loaded_model.eval()  # 设置为评估模式