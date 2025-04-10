import math

import pandas as pd
import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def convert_age(age_str):
    if '以上' in age_str:
        return 70
    elif '-' in age_str:
        start, end = map(int, age_str.replace('岁', '').split('-'))
        return (start + end) // 2


def convert_gender(gender_str):
    if gender_str == '男':
        return 1
    elif gender_str == '女':
        return 2
    else:
        return 0


def convert_fans_cnt(fans_cnt_str):
    fans_cnt_str = str(fans_cnt_str)
    res = 0
    for s in fans_cnt_str:
        if '0' <= s <= '9':
            res = res * 10 + int(s)
    return res


def convert_coin_cnt(coin_cnt_str):
    if coin_cnt_str is None or (isinstance(coin_cnt_str, float) and math.isnan(coin_cnt_str)):
        return 0
    else:
        res = 0
        for s in coin_cnt_str:
            if '0' <= s <= '9':
                res = res * 10 + int(s)
        return res


def convert_post_type(post_type_str):
    if post_type_str == '常规视频':
        return 1
    elif post_type_str == '常规图文':
        return 2
    else:
        return 0


def convert_duration_seconds(duration_str):
    if duration_str is None:
        return 0
    else:
        return int(duration_str)


def data_process(path):
    data = pd.read_csv(path, sep="\t")
    # 将 素材发布时间 与 素材互动量更新时间 做差 得到 统计时长
    # 1. 将这两列转为timestamp格式
    data['update_time'] = pd.to_datetime(data['update_time'], format='%Y%m%d')
    data['publish_time'] = pd.to_datetime(data['publish_time'], format='%Y%m%d')
    # 2. 做差
    data['statistical_duration'] = (data['update_time'] - data['publish_time']).dt.days + 1

    # 将性别处理为 0,1,2
    data['gender'] = data['gender'].apply(convert_gender)
    # 将年龄处理为 int
    data['age'] = data['age'].apply(convert_age)
    # 处理粉丝数量
    data['fans_cnt'] = data['fans_cnt'].apply(convert_fans_cnt)
    # 处理硬币数
    data['coin_cnt'] = data['coin_cnt'].apply(convert_coin_cnt)
    # 处理主帖类型
    data['post_type'] = data['post_type'].apply(convert_post_type)
    # 处理视频时长(数据缺失）
    # data['duration_seconds'] = data['duration_seconds'].apply(convert_duration_seconds)

    return data


if __name__ == '__main__':
    data = data_process('../../Dataset/A/train.txt')
    pd.set_option('display.max_columns', None)  # 显示所有列
    print(data.head(5))
    print(data.shape)