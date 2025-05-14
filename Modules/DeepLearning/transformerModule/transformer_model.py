import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer
from x_transformers import ContinuousTransformerWrapper, ContinuousAutoregressiveWrapper, Decoder
from tqdm import tqdm
from dataProcess import data_process
from torch.nn import MSELoss

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

# 整体特征值 -> int token
X = df[features].values.astype(int)
y = df[target].values
y = np.round(y).astype(int)  # 如果是回归，可以进一步 bin 成分类 token

# 转 Tensor
X_tensor = torch.tensor(X, dtype=torch.long)
y_tensor = torch.tensor(y, dtype=torch.long).unsqueeze(1)  # 作为序列目标


# 超参数
DIM = 512  # 模型维度
ENC_SEQ_LEN = 100  # 编码器输入序列长度（连续特征序列）
DEC_SEQ_LEN = 100  # 解码器输出序列长度（连续特征序列）
DIM_IN = 14  # 输入特征维度（如时间序列的特征数）
DIM_OUT = 14  # 输出特征维度（如预测的连续值维度）
NUM_BATCHES = 1000
LEARNING_RATE = 1e-4
GENERATE_EVERY = 100

# 实例化模型（连续特征编码器-解码器）
model = ContinuousTransformerWrapper(
    dim_in = DIM_IN,               # 输入特征维度
    dim_out = DIM_OUT,             # 输出特征维度
    max_seq_len = ENC_SEQ_LEN,        # 最大序列长度
    attn_layers = Decoder(            # 使用解码器架构（自回归生成）
        dim = DIM,
        depth = 3,
        heads = 8,
        rotary_pos_emb = True,        # 使用旋转位置编码
    )
).cuda()

# 包装为自回归模式（如需生成连续序列）
model = ContinuousAutoregressiveWrapper(model)

# 优化器和损失函数
optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = MSELoss()  # 连续值回归损失（均方误差）


# 模拟数据生成器（返回连续特征张量）
def data_generator():
    while True:
        # 输入：(batch_size, seq_len, dim_in) 连续特征
        src = torch.randn(2, ENC_SEQ_LEN, DIM_IN).cuda()
        # 目标：(batch_size, seq_len, dim_out) 连续值（如未来序列）
        tgt = torch.randn(2, DEC_SEQ_LEN, DIM_OUT).cuda()
        # 掩码（可选，忽略填充位置）
        src_mask = torch.ones(2, ENC_SEQ_LEN).bool().cuda()
        yield src, tgt, src_mask


data_gen = data_generator()

# 训练循环
for i in tqdm(range(NUM_BATCHES), mininterval=10., desc='training'):
    model.train()

    src, tgt, src_mask = next(data_gen)

    # 前向传播（预测 tgt 序列）
    pred = model(src, mask=src_mask)

    # 计算损失（使用均方误差，适合连续值预测）
    loss = torch.nn.functional.mse_loss(pred, tgt)
    loss.backward()
    print(f'{i}: {loss.item()}')

    optim.step()
    optim.zero_grad()

    if i != 0 and i % GENERATE_EVERY == 0:
        model.eval()
        src, _, src_mask = next(data_gen)
        src, src_mask = src[:1], src_mask[:1]  # 取一个样本测试

        # 自回归生成预测序列
        sample = model.generate(src, seq_len=DEC_SEQ_LEN, mask=src_mask)

        # 计算预测误差（均方误差）
        mse_error = torch.nn.functional.mse_loss(sample, src)
        print(f"input shape:  {src.shape}")
        print(f"predicted shape:  {sample.shape}")
        print(f"MSE error: {mse_error.item()}")
