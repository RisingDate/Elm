import numpy as np
import pandas as pd
import torch
from dataProcess import data_process


data_path = '../../Dataset/A/A.txt'
test_data = data_process(data_path)
ids = test_data.iloc[:, 0].values
arr_1d = np.ones(73160)*0.5

combined = np.c_[ids, arr_1d]
print(combined)

# 转换为DataFrame并添加表头
df = pd.DataFrame(combined, columns=["id", "interaction_cnt"])
# 保存为txt文件（tab分隔）
df.to_csv("./results/all05.txt", sep="\t", index=False, header=True)