import pandas as pd

train_path = '../../Dataset/A/train.txt'
B_path = '../../Dataset/B/B.txt'

train_data = pd.read_csv(train_path, sep="\t")
B_data = pd.read_csv(B_path, sep="\t")

uids = train_data['uid'].unique()
matched_uids = B_data['uid'].isin(uids)
num_matched = matched_uids.sum()  # True=1, False=0，求和即匹配的数量

print(f"B_data 中有 {num_matched} 个 uid 出现在 train_data 中")