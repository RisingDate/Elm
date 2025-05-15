import pandas as pd
from sklearn.model_selection import train_test_split

# 读取原始 CSV 文件
file_path = '../../Dataset/A/train.txt'
data = pd.read_csv(file_path, sep='\t')

# 随机分割数据（7:3），并保持随机性可复现（random_state）
train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)

# 保存训练集和测试集到两个文件
train_data.to_csv("../../Dataset/A/train_data.txt", index=False, sep='\t', header=True)
test_data.to_csv("../../Dataset/A/test_data.txt", index=False, sep='\t', header=True)
train_data.to_csv("../../Dataset/A/train_data.csv", index=False, header=True, encoding="utf-8-sig")
test_data.to_csv("../../Dataset/A/test_data.csv", index=False, header=True, encoding="utf-8-sig")

print("训练集大小:", len(train_data))
print("测试集大小:", len(test_data))