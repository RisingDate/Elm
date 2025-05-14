import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertTokenizer
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import time

from dataProcess import CustomFusionDataset,data_process
from model import BertTransformerFusion

# ==== 配置 ====
text_column = ['site_id',]  # 替换成你实际的文本列
numeric_features = [ 'statistical_duration', 'publish_weekday', 'gender', 'age', 'fans_cnt', 'coin_cnt',
                'video_cnt', 'post_type', 'city_level', 'authority_popularity', 'fans_video_ratio', 'avg_coin_per_video',
                'avg_fans_per_video']  # 替换为实际的特征列名
batch_size = 32
num_epochs = 5
lr = 1e-4
path = '../../../Dataset/A/train_data.txt'
# ==== 读取并预处理数据 ====
df = data_process(path) # 修改为你的数据路径
df['interaction_cnt'] = df['interaction_cnt'].apply(lambda x: np.log(x + 1))

# 数值标准化
scaler = StandardScaler()
df[numeric_features] = scaler.fit_transform(df[numeric_features])
joblib.dump(scaler, '../models/fusion-scaler.pkl')

# Tokenizer
tokenizer = BertTokenizer.from_pretrained('./bert-base-chinese')

# Dataset & DataLoader
dataset = CustomFusionDataset(df, tokenizer, text_column, numeric_features)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ==== 模型准备 ====
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BertTransformerFusion(num_features=len(numeric_features)).to(device)

criterion = nn.HuberLoss()
optimizer = optim.AdamW(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

# ==== 训练 ====
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    total_score = 0
    start_time = time.time()

    for input_ids, attention_mask, numeric_feats, targets in loader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        numeric_feats = numeric_feats.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, numeric_feats)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        scheduler.step()

        preds = torch.exp(outputs)
        labels = torch.exp(targets)
        total_score += torch.abs(preds - labels).sum().item()
        total_loss += loss.item()

    print(f'Epoch {epoch+1}/{num_epochs}, '
          f'Loss: {total_loss / len(loader):.4f}, '
          f'Score: {total_score / len(df):.4f}, '
          f'Time: {time.time() - start_time:.2f}s')

# ==== 保存模型 ====
torch.save(model.state_dict(), '../models/fusion-bert-transformer.pth')
