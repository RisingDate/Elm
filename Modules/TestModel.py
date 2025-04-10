import pandas as pd


# 读取CSV到DataFrame
data_A = pd.read_csv('../Dataset/A/train.txt', sep="\t")
pd.set_option('display.max_columns', None)   # 显示所有列

# 查看前5行
print(data_A.head(5))
print(data_A.shape)
# 访问某一列
# print(data_A['column_name'])