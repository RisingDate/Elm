# Elm Baseline讲解

## 解题思考过程

### 问题本质理解

本赛题是典型的回归预测问题，需要根据素材的多种属性（文本内容、作者信息、渠道特征、时间信息等）预测互动量。核心挑战在于：

- 特征多样性：多模态数据（数值、类别、文本、时间）的融合
- 数据时效性：需严格保证特征生成不引入未来信息
- 评价指标：MAE要求误差绝对值最小化

## 常见解题思路

### 数据预处理

#### 基础清洗

- 处理缺失值：对gender、city等字段填充"unknown”，数值字段用均值/中位数填充
- 异常值检测：通过箱线图分析interaction_cnt分布，对极端值进行截断（如取99分位数）
- 时间格式统一：确保publish_time和update_time转为标准datetime格式

#### 关键字段解析

- 作者信息：将birthyear转换为年龄，fans_cnt分段映射为具体数值
- 内容类型：分析post_type(视频/图文）的互动量差异
- 地域分析：解析city字段，提取省份、城市等级（一线/新一线等）

## 特征工程核心方向

### 时间特征

- 基础特征：发布小时、星期几、月份、季度
- 传播周期：update_time - publish_time 的时长
- 时效性特征：节假日标记、周末标记

Tips：互动量 / 传播周期 = 每天的互动量（避免了传播周期的影响）

### 统计特征

- 用户维度：历史平均互动量、最大互动量
- 渠道维度：各渠道平均互动量、内容类型分布
- 交叉统计：用户 * 渠道、 用户 * 内容类型.....的组合统计

Tips：发布数量特别少的用户 和 发布数量特别多的用户加以区分

### 文本特征

- 基础特征：标题/内容长度、特殊符号数量、话题标签数量

- 语义特征：
  - TF-IDF + SVD：提取关键词向量
  - 预训练模型：BERT/ERNIE提取文本Embedding

## 模型选型与优化

### 树模型优先

- CatBoost：自动处理类别变量，适合含大量类别特征的场景
- LightGBM：高效处理大规模数据

### 深度学习尝试

- Wide&Deep：联合训练线性模型和神经网络
- Transformer：对时序特征和文本特征联合建模
- 多模态融合：结构化数据与文本Embedding的拼接

### 模型融合（后期）

- 加权平均：对不同模型的预测结果赋予经验权重
- Stacking：用线性模型组合基模型的预测结果

## Baseline策略

- 分层特征工程：构建基础特征 + 统计特征 + 文本特征的三层体系

- 防泄漏机制：K折交叉验证生成统计特征（5折 -> 4折）

- 鲁棒建模：采用对类别特征友好的CatBoost模型（异常值确定，最后结果/2）
- 后处理优化：保证结果符合业务逻辑（非负整数）

## BaseLine思路

### Baseline模块化设计

- 数据预处理：





## Baseline文件概览

### 数据预处理

<img src="C:\Users\17732\AppData\Roaming\Typora\typora-user-images\image-20250422204253419.png" alt="image-20250422204253419" style="zoom:50%;" />

- 点赞数/粉丝数：僵尸粉丝处理
- 粉丝数/视频数：发一个视频会获得多少粉丝

### 交叉验证特征生成

<img src="C:\Users\17732\AppData\Roaming\Typora\typora-user-images\image-20250422204444491.png" alt="image-20250422204444491" style="zoom:50%;" />

- 防泄漏机制：
  - 训练集每个fold的特征仅用其他fold数据计算
  - 测试集特征使用全量训练数据计算

### 特征工程：

<img src="C:\Users\17732\AppData\Roaming\Typora\typora-user-images\image-20250422204535778.png" alt="image-20250422204535778" style="zoom:50%;" />

- 处理前/后的标签做相关统计（都可以尝试）

- 特征体系：

  | 特征类型 | 示例特征           | 作用         |
  | -------- | ------------------ | ------------ |
  | 基础特征 | 发布时间、粉丝数   | 直接观测值   |
  | 统计特征 | 用户历史平均互动量 | 行为模式捕捉 |
  | 文本特征 | 标题SVD特征        | 内容语义分析 |
  | 组合特征 | 用户 * 渠道统计量  | 交互效应挖掘 |

### 模型训练

<img src="C:\Users\17732\AppData\Roaming\Typora\typora-user-images\image-20250422204702383.png" alt="image-20250422204702383" style="zoom:50%;" />

- 主要是根据离线效果确定模型深度
- 参数设计考量：
  - iterations：充分训练直到早停
  - depth：控制模型复杂度防止过拟合
  - car_features：显示声明类别特征

## Baseline核心逻辑

### 时间衰减特征

<img src="C:\Users\17732\AppData\Roaming\Typora\typora-user-images\image-20250422204821816.png" alt="image-20250422204821816" style="zoom:50%;" />

- 前几天互动量大，后面会慢慢衰减

### 双重统计验证

<img src="C:\Users\17732\AppData\Roaming\Typora\typora-user-images\image-20250422204906288.png" alt="image-20250422204906288" style="zoom:50%;" />

- 

### 文本特征处理

<img src="C:\Users\17732\AppData\Roaming\Typora\typora-user-images\image-20250422204959706.png" alt="image-20250422204959706" style="zoom:50%;" />

<img src="C:\Users\17732\AppData\Roaming\Typora\typora-user-images\image-20250422205015530.png" alt="image-20250422205015530" style="zoom:50%;" />

### 目标优化调整

<img src="C:\Users\17732\AppData\Roaming\Typora\typora-user-images\image-20250422205348588.png" alt="image-20250422205348588" style="zoom:50%;" />
