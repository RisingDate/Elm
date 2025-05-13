import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# Load data
train_df = pd.read_csv(r"D:\study\codes\elm\Elm\Dataset\A\train.txt", sep='\t')
test_df = pd.read_csv(r"D:\study\codes\elm\Elm\Dataset\B\B.txt", sep='\t')
print("Train shape:", train_df.shape, "Test shape:", test_df.shape)

# Drop outliers in interaction_cnt (above 99.8 percentile)
quantile_998 = train_df['interaction_cnt'].quantile(0.998)
train_df = train_df[train_df['interaction_cnt'] <= quantile_998]

# Prepare result DataFrame with id for test set
result_df = test_df[['id']].copy()

# Mark test and train, combine
train_df['istest'] = 0
test_df['istest'] = 1
df = pd.concat([train_df, test_df], ignore_index=True)
del train_df, test_df
df.reset_index(drop=True, inplace=True)

# ========== Categorical feature processing ==========

# Gender: fill missing and create is_gender_known
df['gender'].fillna('未知', inplace=True)
df['is_gender_known'] = df['gender'].apply(lambda g: int(str(g) in ['男', '女']))

# Age: group into categories
def map_age_group(age):
    if age in ['0-5岁', '6-10岁', '11-15岁']:
        return '15岁及以下'
    elif age == '16-20岁':
        return '16-20岁'
    elif age in ['21-25岁', '26-30岁']:
        return '21-30岁'
    elif age == '31-40岁':
        return '31-40岁'
    elif age in ['41-50岁', '51-60岁']:
        return '41-60岁'
    elif age == '60岁以上':
        return '60岁以上'
    else:
        return '未知'
df['age_group'] = df['age'].apply(map_age_group)

# City: map to level
first_tier = ['北京', '上海', '广州', '深圳']
new_first_tier = ['成都', '杭州', '重庆', '武汉', '西安', '苏州', '天津', '南京', '郑州', '长沙',
                  '东莞', '青岛', '合肥', '佛山', '宁波']
second_tier = ['福州', '南昌', '南宁', '昆明', '石家庄', '贵阳', '太原', '常州', '温州', '唐山',
               '烟台', '嘉兴', '南通', '金华', '珠海', '惠州', '徐州', '洛阳', '中山', '台州',
               '兰州', '呼和浩特', '潍坊', '临沂', '绍兴', '廊坊', '扬州']
def map_city_level(city):
    if pd.isna(city) or city == '未知':
        return '其他'
    city_str = str(city)
    for c in first_tier:
        if c in city_str:
            return '一线城市'
    for c in new_first_tier:
        if c in city_str:
            return '新一线城市'
    for c in second_tier:
        if c in city_str:
            return '二线城市'
    return '其他'
df['city_level'] = df['city'].apply(map_city_level)

# Fill and categorize post_type
def fill_post_type(row):
    if pd.notna(row['post_type']):
        return row['post_type']
    # Check video_content for advertising vs normal
    if 'video_content' in row and pd.notna(row['video_content']) and str(row['video_content']).strip():
        if '广告' in str(row.get('title', '')) + str(row.get('content', '')):
            return '广告视频'
        else:
            return '常规视频'
    title = str(row.get('title', ''))
    content = str(row.get('content', ''))
    text = title + content
    # Check for advertisement
    if '广告' in text:
        if '视频' in text:
            return '广告视频'
        elif '图文' in text or '图片' in text:
            return '广告图文'
        else:
            return '广告图文'
    else:
        if '视频' in text:
            return '常规视频'
        elif '图文' in text or '图片' in text:
            return '常规图文'
        else:
            return None

df['post_type'] = df.apply(fill_post_type, axis=1)

def is_ad(post_type):
    if pd.isna(post_type):
        return 0
    return int(str(post_type).startswith('广告'))

def is_video(post_type):
    if pd.isna(post_type):
        return 0
    return int('视频' in str(post_type))

df['is_ad'] = df['post_type'].apply(is_ad)
df['is_video'] = df['post_type'].apply(is_video)

# ========== Numeric feature processing ==========

# Convert fans_cnt, coin_cnt, video_cnt
df['fans_cnt'] = df['fans_cnt'].replace('小于100', 50).fillna(0).astype(int)
df['coin_cnt'] = df['coin_cnt'].replace('小于100', 50).fillna(0).astype(int)
df['video_cnt'] = df['video_cnt'].fillna(0).astype(int)

# Log transform
for col in ['fans_cnt', 'video_cnt', 'coin_cnt']:
    df[f'{col}_log1p'] = np.log1p(df[col])

# Build composite features
df['author_popularity'] = df['coin_cnt'] / (df['fans_cnt'] + 1)
df['fans_video_ratio'] = df['fans_cnt'] / (df['video_cnt'] + 1)
df['author_power'] = np.log1p(df['fans_cnt']) * np.log1p(df['coin_cnt'])
df['avg_coin_per_video'] = df['coin_cnt'] / (df['video_cnt'] + 1)
df['fans_coin_ratio'] = df['fans_cnt'] / (df['coin_cnt'] + 1)
df['video_coin_ratio'] = df['video_cnt'] / (df['coin_cnt'] + 1)
df['video_fans_ratio'] = df['video_cnt'] / (df['fans_cnt'] + 1)
df['log_fans_video_coin'] = np.log1p(df['fans_cnt']) * np.log1p(df['video_cnt']) * np.log1p(df['coin_cnt'])
df['log_sum_fans_video_coin'] = np.log1p(df['fans_cnt'] + df['video_cnt'] + df['coin_cnt'])

# ========== Time feature processing ==========

for col in ['publish_time', 'update_time']:
    df[col] = pd.to_datetime(df[col], format='%Y%m%d')
    df[f'{col}_year'] = df[col].dt.year
    df[f'{col}_month'] = df[col].dt.month
    df[f'{col}_day'] = df[col].dt.day
    df[f'{col}_hour'] = df[col].dt.hour
    df[f'{col}_dayofyear'] = df[col].dt.dayofyear
    df[f'{col}_dayofweek'] = df[col].dt.dayofweek
    df[f'{col}_dayofmonth'] = df[col].dt.day
    df[f'{col}_quarter'] = df[col].dt.quarter
    df[f'{col}_is_weekend'] = df[f'{col}_dayofweek'].apply(lambda x: 1 if x>=5 else 0)
    df[f'{col}_is_month_start'] = df[col].dt.is_month_start.astype(int)
    df[f'{col}_is_month_end'] = df[col].dt.is_month_end.astype(int)

df['time_diff_seconds'] = (df['update_time'] - df['publish_time']).dt.total_seconds()
df['time_diff_days'] = df['time_diff_seconds'] / (24*3600)
df['time_diff_hours'] = df['time_diff_seconds'] / 3600
df['time_diff_minutes'] = df['time_diff_seconds'] / 60
df['time_diff_days'] = df['time_diff_days'].fillna(0).apply(lambda x: max(x, 0))

time_features = []
for col in ['publish_time', 'update_time']:
    for part in ['year','month','day','hour','dayofyear','dayofweek','dayofmonth','quarter','is_weekend','is_month_start','is_month_end']:
        time_features.append(f'{col}_{part}')
time_features += ['time_diff_days', 'time_diff_hours', 'time_diff_minutes']
time_features = [f for f in time_features if f in df.columns]

# ========== Cross features ==========

df['uid'] = df['uid'].astype(str)
df['site_id'] = df['site_id'].astype(str)
df['post_type'] = df['post_type'].fillna('未知').astype(str)
df['age_group'] = df['age_group'].astype(str)
df['city_level'] = df['city_level'].astype(str)
df['is_gender_known'] = df['is_gender_known'].astype(str)

df['user_site'] = df['uid'] + '_' + df['site_id']
df['user_post'] = df['uid'] + '_' + df['post_type']
df['site_post'] = df['site_id'] + '_' + df['post_type']
df['site_age_group'] = df['site_id'] + '_' + df['age_group']
df['site_city_level'] = df['site_id'] + '_' + df['city_level']
df['site_is_gender_known'] = df['site_id'] + '_' + df['is_gender_known']
df['post_age_group'] = df['post_type'] + '_' + df['age_group']
df['post_city_level'] = df['post_type'] + '_' + df['city_level']
df['post_is_gender_known'] = df['post_type'] + '_' + df['is_gender_known']
df['age_group_is_gender_known'] = df['age_group'] + '_' + df['is_gender_known']
df['age_group_city_level'] = df['age_group'] + '_' + df['city_level']
df['is_gender_known_city_level'] = df['is_gender_known'] + '_' + df['city_level']
df['site_post_age_group'] = df['site_id'] + '_' + df['post_type'] + '_' + df['age_group']
df['site_post_city_level'] = df['site_id'] + '_' + df['post_type'] + '_' + df['city_level']

cross_features = [
    'user_site',
    'user_post',
    'site_post',
    'site_age_group',
    'site_city_level',
    'site_is_gender_known',
    'post_age_group',
    'post_city_level',
    'post_is_gender_known',
    'age_group_is_gender_known',
    'age_group_city_level',
    'is_gender_known_city_level',
    'site_post_age_group',
    'site_post_city_level'
]

# ========== KFold target encoding features ==========

def generate_kfold_features(df, target='interaction_cnt', group_key='uid',
                            agg_funcs=['mean', 'max', 'min', 'median'],
                            n_splits=5, random_state=42):
    if target not in df.columns or not pd.api.types.is_numeric_dtype(df[target]):
        return pd.DataFrame(index=df.index)
    train = df[df['istest'] == 0].copy()
    test = df[df['istest'] == 1].copy()
    if len(train) < n_splits:
        return pd.DataFrame(index=df.index)
    folds = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    train['fold'] = -1
    for fold_n, (_, val_idx) in enumerate(folds.split(train)):
        train.loc[train.index[val_idx], 'fold'] = fold_n
    result_df = pd.DataFrame(index=df.index)
    for func in agg_funcs:
        feat_name = f'{group_key}_{target}_{func}'
        train[feat_name] = np.nan
        test[feat_name] = np.nan
        for fold_n in range(n_splits):
            trn_fold = train[train['fold'] != fold_n]
            val_fold = train[train['fold'] == fold_n]
            agg = trn_fold.groupby(group_key)[target].agg(func)
            train.loc[val_fold.index, feat_name] = val_fold[group_key].map(agg)
        full_agg = train.groupby(group_key)[target].agg(func)
        test[feat_name] = test[group_key].map(full_agg)
        result_df[feat_name] = pd.concat([train[feat_name], test[feat_name]], axis=0)
    return result_df

cat_stat_features_list = []
cat_agg_funcs = ['mean', 'max', 'min', 'median']
base_cat_cols = ['site_id', 'is_gender_known', 'age_group', 'city_level', 'post_type']
for col in base_cat_cols:
    if col in df.columns:
        df[col] = df[col].fillna('未知').astype(str)
        feats = generate_kfold_features(df, target='interaction_cnt', group_key=col, agg_funcs=cat_agg_funcs)
        for feat_col in feats.columns:
            df[feat_col] = feats[feat_col]
            cat_stat_features_list.append(feat_col)

cross_stat_features_list = []
cross_agg_funcs = ['mean', 'max', 'min', 'median']
for col in cross_features:
    if col in df.columns:
        df[col] = df[col].fillna('未知_未知').astype(str)
        feats = generate_kfold_features(df, target='interaction_cnt', group_key=col, agg_funcs=cross_agg_funcs)
        for feat_col in feats.columns:
            df[feat_col] = feats[feat_col]
            cross_stat_features_list.append(feat_col)

df['time_diff_days_safe'] = df['time_diff_days'].fillna(0).apply(lambda x: max(x, 1))
df['interaction_cnt_per_day'] = df['interaction_cnt'].fillna(0) / df['time_diff_days_safe']
target_per_day = 'interaction_cnt_per_day'
groups_for_per_day = ['uid', 'user_site', 'user_post']
extra_stat_features_list = []
extra_stat_restore_features_list = []
for group in groups_for_per_day:
    if group in df.columns:
        df[group] = df[group].fillna('未知').astype(str)
        feats = generate_kfold_features(df, target=target_per_day, group_key=group, agg_funcs=cat_agg_funcs)
        for col in feats.columns:
            df[col] = feats[col]
            extra_stat_features_list.append(col)
            restore_col = f'restore_{col}'
            df[restore_col] = df[col] * df['time_diff_days_safe']
            extra_stat_restore_features_list.append(restore_col)

all_stat_features = cat_stat_features_list + cross_stat_features_list + extra_stat_features_list + extra_stat_restore_features_list

df.drop(columns=['time_diff_days_safe', 'interaction_cnt_per_day'], inplace=True, errors='ignore')

# ========== Prepare for model training ==========

train_df_final = df[df['istest'] == 0].copy()
test_df_final = df[df['istest'] == 1].copy()

cat_cols_final = ['site_id', 'gender', 'age', 'city', 'post_type',
                  'is_gender_known', 'age_group', 'city_level', 'is_ad', 'is_video']
numeric_features_final = [
    'fans_cnt', 'video_cnt', 'coin_cnt',
    'fans_cnt_log1p', 'video_cnt_log1p', 'coin_cnt_log1p',
    'avg_coin_per_video', 'fans_coin_ratio', 'video_coin_ratio',
    'video_fans_ratio', 'log_fans_video_coin', 'log_sum_fans_video_coin',
    'author_popularity', 'fans_video_ratio', 'author_power'
]

all_features = cat_cols_final + numeric_features_final + time_features + cross_features + all_stat_features
input_cols = [f for f in all_features if f in df.columns]
input_cols = [f for f in input_cols if f not in ['interaction_cnt', 'istest', 'fold']]

final_cat_feature_names = [f for f in cat_cols_final + cross_features if f in input_cols]
for col in final_cat_feature_names:
    train_df_final[col] = train_df_final[col].fillna('未知').astype(str)
    test_df_final[col] = test_df_final[col].fillna('未知').astype(str)

print("Final feature count:", len(input_cols))
print("Categorical features count:", len(final_cat_feature_names))

# ========== Model training and cross-validation ==========

def cross_validation_train(train_features, target, test_features, cat_features,
                           n_splits=5, max_iterations=10, random_seed_base=42,
                           model_save_dir="models"):
    import os
    os.makedirs(model_save_dir, exist_ok=True)
    models = []
    oof_preds = np.zeros(len(train_features))
    test_preds = np.zeros(len(test_features))
    target_np = target.to_numpy() if isinstance(target, pd.Series) else target
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed_base)
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_features)):
        print(f"Fold {fold+1}/{n_splits}")
        X_train, y_train = train_features.iloc[train_idx], target_np[train_idx]
        X_val, y_val = train_features.iloc[val_idx], target_np[val_idx]
        current_seed = random_seed_base + fold
        model = CatBoostRegressor(
            iterations=max_iterations,
            learning_rate=0.1,
            depth=7,
            loss_function='MAE',
            eval_metric='MAE',
            cat_features=cat_features,
            random_seed=current_seed,
            early_stopping_rounds=200,
            verbose=200
        )
        train_pool = Pool(X_train, y_train, cat_features=cat_features)
        eval_pool = Pool(X_val, y_val, cat_features=cat_features)
        model.fit(train_pool, eval_set=eval_pool)

        # 保存模型
        model_path = os.path.join(model_save_dir, f'catboost_fold{fold}.cbm')
        model.save_model(model_path)
        print(f"Saved model to {model_path}")

        oof_preds[val_idx] = model.predict(X_val)
        test_preds += model.predict(test_features) / n_splits
        models.append(model)
        print(f"Fold {fold+1} MAE: {mean_absolute_error(y_val, oof_preds[val_idx]):.4f}")
    print("OOF MAE:", mean_absolute_error(target_np, oof_preds))
    return models, oof_preds, test_preds

train_target = train_df_final['interaction_cnt'].fillna(0)

models, oof_preds, test_predictions = cross_validation_train(
    train_features=train_df_final[input_cols],
    target=train_target,
    test_features=test_df_final[input_cols],
    cat_features=final_cat_feature_names,
    max_iterations=10
)

# ========== Prediction postprocessing ==========

def postprocess_predictions(preds):
    preds = np.round(preds)
    preds = np.where(preds < 0, 0, preds)
    return preds.astype(int)

result_df['interaction_cnt'] = postprocess_predictions(test_predictions)
print("Average prediction:", result_df['interaction_cnt'].mean())

scale_factor = 1.0
result_df['interaction_cnt'] = np.round(result_df['interaction_cnt'] * scale_factor).astype(int)
result_df['interaction_cnt'] = np.where(result_df['interaction_cnt'] < 0, 0, result_df['interaction_cnt'])
print("Average after scaling:", result_df['interaction_cnt'].mean())

# ========== Save results ==========

import os
if not os.path.exists('results'):
    os.makedirs('results')
timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
output_filename = f"results/final_results_{timestamp}_scale{scale_factor:.2f}.csv"
result_df[['id', 'interaction_cnt']].to_csv(output_filename, index=False)
print(f"Results saved to {output_filename}")

# ========== Feature importance ==========

def get_feature_importance(model, feature_names):
    try:
        importance = model.get_feature_importance()
        return pd.DataFrame({'feature': feature_names, 'importance': importance}).sort_values('importance', ascending=False)
    except Exception as e:
        print("Feature importance error:", e)
        return pd.DataFrame(columns=['feature', 'importance'])

if models:
    importance_df = get_feature_importance(models[0], input_cols)
    print("\nFeature importance (first model):")
    print(importance_df.to_string(index=False))
