import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.preprocessing import StandardScaler
import joblib
import time
from dataProcess import data_process, CustomDataset

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------- 数据处理 ----------------------


def row_to_text(row, feature_names):
    return ', '.join([f"{name}: {row[name]}" for name in feature_names])


class BertTextDataset(Dataset):
    def __init__(self, df, feature_cols, target_col, tokenizer, max_len=128):
        self.tokenizer = tokenizer
        self.texts = [row_to_text(row, feature_cols) for _, row in df.iterrows()]
        self.labels = np.log(df[target_col].values + 1)

        self.encodings = self.tokenizer(
            self.texts,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors='pt'
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float32)
        return item


# ---------------------- 模型定义 ----------------------
class BertRegressor(nn.Module):
    def __init__(self, bert_model_name='bert-base-chinese', hidden_dim=128):
        super(BertRegressor, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.regressor = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.pooler_output  # shape: (batch_size, hidden_size)
        return self.regressor(cls_output).squeeze(1)  # shape: (batch_size,)


# ---------------------- 主训练流程 ----------------------
if __name__ == '__main__':
    path = '../../../Dataset/A/train_data.txt'
    df = data_process(path)

    feature_columns = ['site_id', 'statistical_duration', 'publish_weekday', 'gender', 'age',
                       'fans_cnt', 'coin_cnt', 'video_cnt', 'post_type', 'city_level',
                       'authority_popularity', 'fans_video_ratio', 'avg_coin_per_video', 'avg_fans_per_video']
    target_column = 'interaction_cnt'

    tokenizer = BertTokenizer.from_pretrained('./bert-base-chinese')
    dataset = BertTextDataset(df, feature_columns, target_column, tokenizer)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = BertRegressor().to(device)
    criterion = nn.HuberLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-6)

    num_epochs = 5
    model.train()

    for epoch in range(num_epochs):
        epoch_start = time.time()
        total_loss, total_score = 0.0, 0.0

        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            # 还原为原始尺度计算绝对误差
            original_labels = torch.exp(labels)
            original_outputs = torch.exp(outputs)
            abs_diff = torch.abs(original_labels - original_outputs)
            total_score += abs_diff.sum().item()
            total_loss += loss.item()

        print(f"[Epoch {epoch+1}] Loss: {total_loss / len(dataloader):.4f}, "
              f"Score: {total_score / len(dataset):.4f}, "
              f"Time: {time.time() - epoch_start:.2f}s")

    # 保存模型
    torch.save(model.state_dict(), '../models/bert-regressor.pth')
