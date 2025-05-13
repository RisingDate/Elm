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