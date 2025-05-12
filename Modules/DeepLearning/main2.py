import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from Modules.DeepLearning.run.model import InteractionPredictor
from Modules.DeepLearning.run.dataProcess import data_process


def prepare_data(train_data):
    # 区分数值和类别特征
    numeric_features = ['statistical_duration', 'age', 'fans_cnt', 'coin_cnt']
    categorical_features = ['site_id', 'gender', 'post_type']

    # 创建预处理管道
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    X = preprocessor.fit_transform(train_data[features])
    y = np.log(train_data['interaction_cnt'].values + 1)

    return X, y, preprocessor


if __name__ == '__main__':
    path = '../../Dataset/A/train.txt'
    train_data = data_process(path)
    features = ['site_id', 'statistical_duration', 'gender', 'age', 'fans_cnt', 'coin_cnt', 'post_type']

    # 数据预处理
    X, y, preprocessor = prepare_data(train_data)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # 转换为PyTorch张量
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)

    # 创建数据加载器
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = InteractionPredictor(X_train.shape[1]).to(device)

    # 定义损失函数和优化器
    criterion = nn.HuberLoss()  # 对异常值更鲁棒
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    # 训练循环
    best_val_loss = float('inf')
    patience = 10
    no_improve = 0

    for epoch in range(100):
        model.train()
        train_loss = 0
        train_mae = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 计算训练指标
            with torch.no_grad():
                pred = torch.exp(outputs) - 1
                true = torch.exp(labels) - 1
                mae = torch.abs(pred - true).mean()

            train_loss += loss.item()
            train_mae += mae.item()

        # 验证阶段
        model.eval()
        val_loss = 0
        val_mae = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                pred = torch.exp(outputs) - 1
                true = torch.exp(labels) - 1
                mae = torch.abs(pred - true).mean()

                val_loss += loss.item()
                val_mae += mae.item()

        # 计算平均指标
        train_loss /= len(train_loader)
        train_mae /= len(train_loader)
        val_loss /= len(val_loader)
        val_mae /= len(val_loader)

        # 学习率调整
        scheduler.step(val_loss)

        # 早停机制
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        print(f"Epoch {epoch + 1}: "
              f"Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.2f} | "
              f"Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.2f}")