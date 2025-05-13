import os
os.chdir(os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
import math

import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from ..embeddingText import process_text_features


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


def convert_fans_cnt(row):
    x = row['fans_cnt']
    coin_cnt = row['coin_cnt']
    if x == "" or (isinstance(x, float) and math.isnan(x)):
        if not math.isnan(coin_cnt):
            print(coin_cnt)
            return int(coin_cnt / 15)
        else:
            return 0
    try:
        return float(x)
    except (ValueError, TypeError):
        if x == "小于100":
            return 50
        else:
            return x


def convert_coin_cnt(row):
    x = row['coin_cnt']
    fans_cnt = row['fans_cnt']
    if x == "" or (isinstance(x, float) and math.isnan(x)):
        if fans_cnt != "" and (isinstance(fans_cnt, float) and math.isnan(fans_cnt)) is False:
            return fans_cnt * 15
        else:
            return 0
    try:
        return float(x)          # 尝试转换为数值
    except (ValueError, TypeError):
        if x == "小于100":
            return 50
        else:
            return x


def convert_video_cnt(x):
    if x == "" or (isinstance(x, float) and math.isnan(x)):
        return 0
    else:
        return float(x)


def convert_post_type(row):
    # 常规视频1, 常规图文2， 广告视频3， 广告图文4， 其他5
    post_type_str = row['post_type']
    # 判断为 nan
    if post_type_str is None or post_type_str == "" or pd.isna(post_type_str):
        if 'video_content' in row and pd.notna(row['video_content']) and str(row['video_content']).strip():
            if '广告' in str(row.get('title', '')) + str(row.get('content', '')):
                return 3
            else:
                return 1
        else:
            title = str(row.get('title', ''))
            content = str(row.get('content', ''))
            text = title + content
            if '广告' in text:
                if '视频' in text:
                    return 3
                else:
                    return 4
            else:
                if '视频' in text:
                    return 1
                elif '图文' in text or '图片' in text:
                    return 2
                else:
                    return 5
    if post_type_str == '常规视频':
        return 1
    elif post_type_str == '常规图文':
        return 2
    elif post_type_str == '广告视频':
        return 3
    elif post_type_str == '广告图文':
        return 4
    else:
        return 5


city_first_tier = ['北京', '上海', '广州', '深圳']
city_new_first_tier = ['成都', '重庆', '杭州', '武汉', '苏州',
                       '西安', '南京', '长沙', '天津', '郑州',
                       '东莞', '青岛', '沈阳', '宁波', '昆明']
city_second_tier = ['厦门', '福州', '济南', '合肥', '无锡',
                    '常州', '温州', '绍兴', '泉州', '嘉兴',
                    '金华', '烟台', '珠海', '中山', '惠州',
                    '海口', '南昌', '太原', '洛阳', '南宁',
                    '贵阳', '遵义', '兰州', '乌鲁木齐',
                    '银川', '大连', '哈尔滨', '长春']


def convert_city(city_str):
    if city_str is None or city_str == "" or pd.isna(city_str):
        return 3
    else:
        if any(keyword in city_str for keyword in city_first_tier):
            return 0
        elif any(keyword in city_str for keyword in city_new_first_tier):
            return 1
        elif any(keyword in city_str for keyword in city_second_tier):
            return 2
        else:
            return 3


def convert_duration_seconds(duration_str):
    if duration_str is None:
        return 0
    else:
        return int(duration_str)


def data_process(path, is_train=True):
    data = pd.read_csv(path, sep="\t")
    # 训练集去除数据中的极端点 -> 从train中剔除interaction_cnt 99.8%分位数以上的数据
    if is_train:
        quantile_998 = data['interaction_cnt'].quantile(0.998)
        print(f'quantile_998: {quantile_998}')
        data = data[data['interaction_cnt'] <= quantile_998]

    # 将 素材发布时间 与 素材互动量更新时间 做差 得到 统计时长
    # 1. 将这两列转为timestamp格式
    data['update_time'] = pd.to_datetime(data['update_time'], format='%Y%m%d')
    data['publish_time'] = pd.to_datetime(data['publish_time'], format='%Y%m%d')
    # 2. 做差
    data['statistical_duration'] = (data['update_time'] - data['publish_time']).dt.days + 1
    # 3. 周几
    data['publish_weekday'] = data['publish_time'].dt.weekday >= 5

    # 将性别处理为 0,1,2
    data['gender'] = data['gender'].apply(convert_gender)
    # 将年龄处理为 int
    data['age'] = data['age'].apply(convert_age)
    # 处理粉丝数量
    data['fans_cnt'] = data.apply(convert_fans_cnt, axis=1)
    # 处理硬币数
    data['coin_cnt'] = data.apply(convert_coin_cnt, axis=1)
    # 处理视频数量
    data['video_cnt'] = data['video_cnt'].apply(convert_video_cnt)
    # 处理城市等级（一线，新一线，二线，其他）
    data['city_level'] = data['city'].apply(convert_city)
    # 处理主帖类型
    data['post_type'] = data.apply(convert_post_type, axis=1)
    # 文本特征处理
    text_columns = [col for col in ['title', 'content', 'cover_ocr_content', 'video_content'] if col in data.columns]
    if text_columns:
        text_features_df, text_feature_names = process_text_features(data, text_columns=text_columns)
        data = pd.concat([data.reset_index(drop=True), text_features_df.reset_index(drop=True)], axis=1)

    # 交叉特征
    data['authority_popularity'] = data['coin_cnt'] / (data['fans_cnt'] + 1)
    data['fans_video_ratio'] = data['fans_cnt'] / (data['video_cnt'] + 1)
    # 作品均获硬币数
    data['avg_coin_per_video'] = data['coin_cnt'] / (data['video_cnt'] + 1)
    # 作品均获粉丝数
    data['avg_fans_per_video'] = data['fans_cnt'] / (data['video_cnt'] + 1)
    return data


if __name__ == '__main__':
    data = data_process('../../../Dataset/A/train.txt')
    features = ['site_id', 'statistical_duration', 'fans_cnt', 'coin_cnt']
    data = data[features]
    pd.set_option('display.max_columns', None)  # 显示所有列
    print(data.head(50))
    scaler = StandardScaler()
    x_train = scaler.fit_transform(data.values)
    print(x_train[:50])
