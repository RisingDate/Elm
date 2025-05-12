import math

import numpy as np
import pandas as pd
import torch
from dataProcess import data_process


def process_value(x):
    if x == "" or (isinstance(x, float) and math.isnan(x)):
        return 0
    try:
        x_num = float(x)          # 尝试转换为数值
        return 999 if x_num >= 100 else x_num
    except (ValueError, TypeError):
        if x == "小于100":
            return 50
        else:
            return x


def convert_gender(gender_str):
    if gender_str == '男':
        return 1
    elif gender_str == '女':
        return 2
    else:
        return 0


def convert_age(age_str):
    if '以上' in age_str:
        return 70
    elif '-' in age_str:
        start, end = map(int, age_str.replace('岁', '').split('-'))
        return (start + end) // 2


def convert_city(city_str):
    if city_str is None or city_str == "" or pd.isna(city_str):
        return '未知'
    else:
        return city_str


data_path = '../../Dataset/B/B.txt'
data = pd.read_csv(data_path, sep="\t")

data['fans_cnt'] = data['fans_cnt'].apply(process_value)
unique_fans_cnt = data['fans_cnt'].unique()
print('unique_fans_cnt', unique_fans_cnt)
# pd.DataFrame(unique_fans_cnt, columns=['fans_cnt']).to_csv('../../Dataset/A/test/unique_fans_cnt.csv', index=False)

unique_site_id = data['site_id'].unique()
print('unique_site_id', unique_site_id)
# pd.DataFrame(unique_site_id, columns=['site_id']).to_csv('../../Dataset/A/test/unique_site_id.csv', index=False)

unique_age = data['age'].apply(convert_age).unique()
print('unique_age', unique_age)
# pd.DataFrame(unique_age, columns=['age']).to_csv('../../Dataset/A/test/unique_age.csv', index=False)

unique_gender = data['gender'].apply(convert_gender).unique()
print('unique_gender', unique_gender)

unique_city = data['city'].apply(convert_city).unique()
print('unique_city', unique_city)
pd.DataFrame(unique_city, columns=['city']).to_csv('../../Dataset/B/test/unique_city.csv', index=False)

citys = data['city'].apply(convert_city)
pd.DataFrame({
    'city': citys,
    'interaction_cnt': data['interaction_cnt']
}).to_csv('../../Dataset/B/test/city_and_interaction.csv', index=False)