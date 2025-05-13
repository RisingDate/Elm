import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer
from x_transformers import ContinuousTransformerWrapper
import tqdm
from dataProcess import data_process

# ========== 数据读取与预处理 ==========

# 特征字段
features = [
    'site_id', 'statistical_duration', 'publish_weekday', 'gender', 'age',
    'fans_cnt', 'coin_cnt', 'video_cnt', 'post_type', 'city_level',
    'authority_popularity', 'fans_video_ratio', 'avg_coin_per_video', 'avg_fans_per_video'
]
target = 'interaction_cnt'

file_path = '../../../Dataset/A/train_data.txt'
# 加载数据
df = data_process(file_path)
# df = pd.read_csv("your_data.csv")  # << 替换为你的数据路径

# 分类特征编码（LabelEncoder）
cat_features = ['site_id', 'post_type', 'city_level']
label_encoders = {col: LabelEncoder().fit(df[col]) for col in cat_features}
for col in cat_features:
    df[col] = label_encoders[col].transform(df[col])

# 连续特征离散化（KBinsDiscretizer）
num_features = [f for f in features if f not in cat_features]
binner = KBinsDiscretizer(n_bins=16, encode='ordinal', strategy='uniform')
df[num_features] = binner.fit_transform(df[num_features])

# 整体特征值 -> int token
X = df[features].values.astype(int)
y = df[target].values
y = np.round(y).astype(int)  # 如果是回归，可以进一步 bin 成分类 token

# 转 Tensor
X_tensor = torch.tensor(X, dtype=torch.long)
y_tensor = torch.tensor(y, dtype=torch.long).unsqueeze(1)  # 作为序列目标

# Dataset 和 Dataloader
BATCH_SIZE = 16
dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


# 数据迭代器（替代 cycle()）
def cycle_data():
    while True:
        for xb, yb in dataloader:
            src = xb.cuda()  # [batch, seq_len]
            tgt = yb.cuda()  # [batch, 1]
            src_mask = torch.ones(src.shape[0], src.shape[1], dtype=torch.bool).cuda()
            yield src, tgt, src_mask


data_iter = cycle_data()

# ========== 模型配置 ==========

NUM_TOKENS = 16 + 2  # 离散化后最大 token 值 + 特殊 token
ENC_SEQ_LEN = len(features)
DEC_SEQ_LEN = 1  # 只预测 interaction_cnt
LEARNING_RATE = 3e-4
NUM_BATCHES = 1000
GENERATE_EVERY = 100

# 模型实例
model = XTransformer(
    dim=512,
    tie_token_emb=True,
    return_tgt_loss=True,
    enc_num_tokens=NUM_TOKENS,
    enc_depth=3,
    enc_heads=8,
    enc_max_seq_len=ENC_SEQ_LEN,
    dec_num_tokens=NUM_TOKENS,
    dec_depth=3,
    dec_heads=8,
    dec_max_seq_len=DEC_SEQ_LEN
).cuda()

# 优化器
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ========== 训练循环 ==========

for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=10., desc='training'):
    model.train()
    src, tgt, src_mask = next(data_iter)
    if src.max().item() > NUM_TOKENS:
        print("=========src max token===========", src.max().item())  # 应 < NUM_TOKENS
    if tgt.max().item() > NUM_TOKENS:
        print("=========tgt max token===========", tgt.max().item())  # 应 < NUM_TOKENS
    # print("src max token:", src.max().item())
    # print("tgt max token:", tgt.max().item())
    # print("src shape:", src.shape, "tgt shape:", tgt.shape)

    loss = model(src, tgt, mask=src_mask)
    # print('test', loss)
    loss.backward()
    print(f'{i}: loss = {loss.item()}')
    optimizer.step()
    optimizer.zero_grad()

    # 生成检查
    if i != 0 and i % GENERATE_EVERY == 0:
        model.eval()
        src, _, src_mask = next(data_iter)
        src, src_mask = src[:1], src_mask[:1]
        start_tokens = (torch.ones((1, 1)) * 1).long().cuda()  # 起始 token

        sample = model.generate(src, start_tokens, DEC_SEQ_LEN, mask=src_mask)
        print(f"input: {src}")
        print(f"predicted output: {sample}")
        print(f"target: {tgt[:1]}")
        incorrects = (tgt[:1] != sample).sum()
        print(f"incorrects: {incorrects.item()}")
