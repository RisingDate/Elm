# model介绍

## model 1

**神经网络：**InteractionPredictor

**优化器：**optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

**损失函数：**criterion = nn.HuberLoss()

**特征列：**features = ['site_id', 'statistical_duration', 'gender', 'age', 'fans_cnt', 'coin_cnt', 'post_type']

**损失函数值：**Loss: 0.34164165988126904


**得分：**Score: 160.89742465943505

**对应输出文件：**output-250512-1.txt

**提交得分：**39.8760



## model 2

**神经网络：**InteractionPredictor

**优化器：**optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

**损失函数：**criterion = nn.HuberLoss()

**特征列：**features = ['site_id', 'statistical_duration', 'publish_weekday', 'gender', 'age', 'fans_cnt', 'coin_cnt', 'post_type']

**损失函数值：**Loss: 0.33378298287138797

**得分：**Score：160

**对应输出文件：**output-250512-2.txt

**提交得分：**



## model 3

**神经网络：**InteractionPredictor

**优化器：**optim.NAdam(model.parameters(), lr=0.001)

**损失函数：**LogCoshLoss()

**特征列：**features = ['site_id', 'statistical_duration', 'publish_weekday', 'gender', 'age', 'fans_cnt', 'coin_cnt', 'post_type']

**损失函数值：**Loss: 0.3061109164224048

**得分：**Score: 162.48111342701284

**对应输出文件：**output-250512-3.txt（超出数据范围）

**提交得分：**



## model 4

**神经网络：**EnhancedInteractionPredictor

**优化器：**optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

**损失函数：**nn.HuberLoss()

**特征列：**features = ['site_id', 'statistical_duration', 'publish_weekday', 'gender', 'age', 'fans_cnt', 'coin_cnt', 'post_type']

**损失函数值：**Loss: 0.3411730793943371

**得分：**Score: 198.85225268358127（不稳定）

**对应输出文件：**output-250512-4.txt（超出数据范围）

**提交得分：**



## model 5

**神经网络：**EnhancedInteractionPredictor

**优化器：**optim.NAdam(model.parameters(), lr=0.001)

**损失函数：**LogCoshLoss()

**学习率调度器：**optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)

**特征列：**features = ['site_id', 'statistical_duration', 'publish_weekday', 'gender', 'age', 'fans_cnt', 'coin_cnt', 'post_type']	Ps：weekday修改为是否为周末

**损失函数值：**Loss: 0.3074601068056978

**得分：**Score: 166.95452076042267

**对应输出文件：**output-250512-5.txt（超出数据范围）

**提交得分：**



## model 6

**优化器：**optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

**损失函数：**criterion = nn.HuberLoss()

**特征列：**features = ['site_id', 'statistical_duration', 'publish_weekday', 'gender', 'age', 'fans_cnt', 'coin_cnt', 'post_type']

**损失函数值：**

**得分：**

**对应输出文件：**output-250513-1.txt

**提交得分：** 35.5024

**Ps：**剔除噪音数据（从train中剔除interaction_cnt 99.8%分位数以上的数据）



## model 7

**优化器：**optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

**损失函数：**criterion = nn.HuberLoss()

**特征列：**features = ['site_id', 'statistical_duration', 'publish_weekday', 'gender', 'age', 'fans_cnt', 'coin_cnt', 'video_cnt', 'post_type']

**损失函数值：**

**得分：**

**对应输出文件：**

**提交得分：**



## model 8

**优化器：**criterion = LogCoshLoss()

**损失函数：**optim.NAdam(model.parameters(), lr=0.001)

**特征列：**

```
features = ['site_id', 'statistical_duration', 'publish_weekday', 'gender', 'age', 'fans_cnt', 'coin_cnt',
            'video_cnt', 'post_type', 'city_level', 'authority_popularity', 'fans_video_ratio', 'avg_coin_per_video',
            'avg_fans_per_video']
```

**损失函数值：**Loss: 0.32943986714679124

**得分：**Score: 33.19924349409636

**对应输出文件：**output-250513-2.txt

**提交得分：**



## model 9

**优化器：**optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

**损失函数：**criterion = nn.HuberLoss()

**特征列：**

```
features = ['site_id', 'statistical_duration', 'publish_weekday', 'gender', 'age', 'fans_cnt', 'coin_cnt',
                'video_cnt', 'post_type', 'city_level', 'authority_popularity', 'fans_video_ratio', 'avg_coin_per_video',
                'avg_fans_per_video']
```

**损失函数值：**Loss: 0.2972138170508525

**得分：**Score: 33.51822020559602

**对应输出文件：**output-250513-3.txt

**提交得分：**141.2515



## model 10

**优化器：**optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

**损失函数：**criterion = nn.HuberLoss()

**特征列：**

```
features = ['site_id', 'statistical_duration', 'fans_cnt', 'coin_cnt', 'video_cnt', 'post_type',
            'authority_popularity', 'fans_video_ratio', 'avg_coin_per_video', 'avg_fans_per_video']
```

**损失函数值：**

**得分：**

**对应输出文件：**output-250513-4.txt

**提交得分：**



## model 11

**模型：**EnhancedInteractionPredictor

**优化器：**optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)

**损失函数：**criterion = nn.HuberLoss()

**特征列：**

```
features = ['site_id', 'statistical_duration', 'publish_weekday', 'gender', 'age', 'fans_cnt', 'coin_cnt',
            'video_cnt', 'post_type', 'city_level', 'authority_popularity', 'fans_video_ratio', 'avg_coin_per_video',
            'avg_fans_per_video']
```

**损失函数值：**

**得分：**

**对应输出文件：**

**提交得分：**



## tf-model 1

**模型：**XTransformer(input_dim=x_train.shape[1], dim=64, depth=4, heads=4)

**优化器：**optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

**损失函数：**criterion = nn.HuberLoss()

**特征列：**

```
features = ['site_id', 'statistical_duration', 'publish_weekday', 'gender', 'age', 'fans_cnt', 'coin_cnt', 'video_cnt', 'post_type', 'city_level', 'authority_popularity', 'fans_video_ratio', 'avg_coin_per_video', 'avg_fans_per_video']
```

**损失函数值：**

**得分：**Score: 28.267368204800437

**对应输出文件：**output-250514-2-tf

**提交得分：**



## tf-model 2

**模型：**XTransformer(input_dim=x_train.shape[1], dim=64, depth=4, heads=4)

**优化器：**optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

**损失函数：**criterion = nn.HuberLoss()

**特征列：**

```
features = ['site_id', 'statistical_duration', 'fans_cnt', 'coin_cnt',
            'video_cnt', 'post_type', 'authority_popularity', 'fans_video_ratio',
            'avg_coin_per_video', 'avg_fans_per_video']
```

**损失函数值：**Loss: 0.3183

**得分：**Score: 28.562219660568154

**对应输出文件：**

**提交得分：**



## tf-model 3

**模型：**XTransformer(input_dim=x_train.shape[1], dim=64, **depth=8**, heads=4)

**优化器：**optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

**损失函数：**criterion = nn.HuberLoss()

**特征列：**

```
features = ['site_id', 'statistical_duration', 'publish_weekday', 'gender', 'age', 'fans_cnt', 'coin_cnt', 'video_cnt', 'post_type', 'city_level', 'authority_popularity', 'fans_video_ratio', 'avg_coin_per_video', 'avg_fans_per_video']
```

**损失函数值：**Loss: 0.313083

**得分：**Score: 28.29616969309609

**对应输出文件：**output-250514-4-tf.txt

**提交得分：**



## tf-model 4

**模型：**XTransformer(input_dim=x_train.shape[1], dim=64, **depth=8**, heads=4)

**优化器：**optimizer = optim.NAdam(model.parameters(), lr=0.001)

**损失函数：**criterion = LogCoshLoss()

**特征列：**

```
features = ['site_id', 'statistical_duration', 'publish_weekday', 'gender', 'age', 'fans_cnt', 'coin_cnt',
                'video_cnt', 'post_type', 'city_level', 'authority_popularity', 'fans_video_ratio', 'avg_coin_per_video',
                'avg_fans_per_video']
```

**损失函数值：**Loss: 0.27730889

**得分：**Score: 28.579653491902256

**对应输出文件：**

**提交得分：**



## tf-model 5

**模型：**XTransformer(input_dim=x_train.shape[1], dim=64, depth=4, heads=4)

**优化器：**optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

**损失函数：**criterion = nn.HuberLoss()

**特征列：**

```
str_features = ['user_site', 'user_post', 'site_post', 'site_age_group']
features = ['site_id', 'statistical_duration', 'publish_weekday', 'gender', 'age', 'fans_cnt', 'coin_cnt',
            'video_cnt', 'post_type', 'city_level', 'authority_popularity', 'fans_video_ratio', 'avg_coin_per_video',
            'avg_fans_per_video']
```

**损失函数值：**Loss: 0.0972

**得分：**Score: 19.1966

**对应输出文件：**

**提交得分：**



## tf-model 6

**模型：**XTransformer(input_dim=x_train.shape[1], dim=64, **depth=10**, heads=4)

**优化器：**optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

**损失函数：**criterion = nn.HuberLoss()

**特征列：**

```
features = ['site_id', 'statistical_duration', 'publish_weekday', 'gender', 'age', 'fans_cnt', 'coin_cnt',
                'video_cnt', 'post_type', 'city_level', 'authority_popularity', 'fans_video_ratio', 'avg_coin_per_video',
                'avg_fans_per_video', 'user_site', 'user_post', 'site_post', 'site_age_group']
```

**损失函数值：**

**得分：**Score: 28.286650992387322

**对应输出文件：**

**提交得分：**



## **tf-model 7

**模型：**

```
model = XTransformerWithEmbedding(
    num_numeric_features=len(numeric_features),
    categorical_info=categorical_info,
    embed_dim=8,
    dim=64,
    depth=8,
    heads=4,
    dropout=0.1
)
```

**优化器：**optimizer = optim.NAdam(model.parameters(), lr=0.001)

**损失函数：**criterion = LogCoshLoss()

**特征列：**

```
numeric_features = ['site_id', 'statistical_duration', 'publish_weekday', 'gender', 'age', 'fans_cnt', 'coin_cnt',
                    'video_cnt', 'post_type', 'city_level', 'authority_popularity', 'fans_video_ratio',
                    'avg_coin_per_video', 'avg_fans_per_video', 'site_post', 'site_age_group', 'site_city']
epoch = 200
```

**损失函数值：**Loss: 0.26350318

**得分：**27.751429517102846

**对应输出文件：**

**提交得分：**



## tf-model 8

**模型：**

```
model = XTransformerWithEmbedding(
    num_numeric_features=len(numeric_features),
    categorical_info=categorical_info,
    embed_dim=8,
    dim=64,
    depth=4,
    heads=4,
    dropout=0.1
)
```

**优化器：**optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

**损失函数：**criterion = nn.HuberLoss()

**特征列：**

```
str_features = ['user_site', 'user_post', 'uid']
numeric_features = ['site_id', 'statistical_duration', 'publish_weekday', 'gender', 'age', 'fans_cnt', 'coin_cnt',
                        'video_cnt', 'post_type', 'city_level', 'authority_popularity', 'fans_video_ratio',
                        'avg_coin_per_video', 'avg_fans_per_video', 'site_post', 'site_age_group', 'site_city']
```

**损失函数值：**

**得分：**

**对应输出文件：**

**提交得分：**



## model-with-2-feature

**优化器：**optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

**损失函数：**criterion = nn.HuberLoss()

**特征列：**features = ['fans_cnt', 'coin_cnt']

**损失函数值：**

**得分：**

**提交得分：**



## model n (Templete)

**优化器：**

**损失函数：**

**特征列：**

**损失函数值：**

**得分：**

**对应输出文件：**

**提交得分：**





```

```

