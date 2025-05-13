import math

import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns

matplotlib.use('TkAgg')


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
        x_num = float(x)  # 尝试转换为数值
        return 999 if x_num >= 100 else x_num
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
        x_num = float(x)  # 尝试转换为数值
        return 999 if x_num >= 100 else x_num
    except (ValueError, TypeError):
        if x == "小于100":
            return 50
        else:
            return x


def convert_video_cnt(x):
    if x == "" or (isinstance(x, float) and math.isnan(x)):
        return 0
    else:
        x_num = float(x)
        return 999 if x_num >= 100 else x_num


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


data_path = '../../Dataset/A/train_data.txt'
data = pd.read_csv(data_path, sep="\t")

data['fans_cnt'] = data.apply(convert_fans_cnt, axis=1)
unique_fans_cnt = data['fans_cnt'].unique()
print('unique_fans_cnt', unique_fans_cnt)
# pd.DataFrame(unique_fans_cnt, columns=['fans_cnt']).to_csv('../../Dataset/A/test/unique_fans_cnt.csv', index=False)

data['coin_cnt'] = data.apply(convert_coin_cnt, axis=1)
unique_coin_cnt = data['coin_cnt'].unique()
print('unique_coin_cnt', unique_coin_cnt)

unique_site_id = data['site_id'].unique()
print('unique_site_id', unique_site_id)
# pd.DataFrame(unique_site_id, columns=['site_id']).to_csv('../../Dataset/A/test/unique_site_id.csv', index=False)

unique_age = data['age'].apply(convert_age).unique()
print('unique_age', unique_age)
# pd.DataFrame(unique_age, columns=['age']).to_csv('../../Dataset/A/test/unique_age.csv', index=False)

unique_video_cnt = data['video_cnt'].apply(convert_video_cnt).unique()
print('unique_video_cnt', unique_video_cnt)

unique_gender = data['gender'].apply(convert_gender).unique()
print('unique_gender', unique_gender)

unique_city = data['city'].apply(convert_city).unique()
print('unique_city', unique_city)

unique_post_type = data.apply(convert_post_type, axis=1).unique()
print('post_type', unique_post_type)
# pd.DataFrame(unique_city, columns=['city']).to_csv('../../Dataset/B/test/unique_city.csv', index=False)

# citys = data['city'].apply(convert_city)
# pd.DataFrame({
#     'city': citys,
#     'interaction_cnt': data['interaction_cnt']
# }).to_csv('../../Dataset/B/test/city_and_interaction.csv', index=False)
