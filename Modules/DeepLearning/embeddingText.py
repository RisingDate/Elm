import os
from openai import OpenAI
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

def get_text_embeddings(file_path, column_name="content", dim=1024, batch_size=32):
    """
    读取指定文件的指定列，获取文本embedding。
    :param file_path: 文件路径（如B.txt）
    :param column_name: 需要embedding的列名
    :param dim: embedding维度
    :param batch_size: 批处理大小
    :return: embedding的DataFrame，每行对应一个文本的embedding
    """
    client = OpenAI(
        api_key="sk-25f781189f5a4a2cb98cebac77ae80d4",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    df = pd.read_csv(file_path, sep="\t")
    texts = df[column_name].fillna("").tolist()
    all_embs = []
    for i in range(0, len(texts), batch_size):
        resp = client.embeddings.create(
            model="text-embedding-v3",
            input=texts[i:i+batch_size],
            dimensions=dim,
            encoding_format="float"
        )
        all_embs.extend([item.embedding for item in resp.data])
    emb_df = pd.DataFrame(all_embs, columns=[f"emb_{i}" for i in range(dim)])
    return emb_df

# In [55]: 文本特征处理函数
def process_text_features(df, text_columns=['title', 'content', 'cover_ocr_content', 'video_content'], n_components=50, max_features_tfidf=5000):
    """使用 TF-IDF 和 TruncatedSVD 处理文本列"""
    print("开始处理文本特征...")
    all_text_features = []
    final_text_feature_names = []
 
    for col in text_columns:
        print(f"处理列: {col}")
        if col not in df.columns:
            print(f"警告: 列 {col} 不存在，跳过。")
            continue
 
        # 填充缺失值为空字符串
        df[col] = df[col].fillna('')
 
        # TF-IDF
        print(f"  计算 TF-IDF (max_features={max_features_tfidf})...")
        tfidf = TfidfVectorizer(max_features=max_features_tfidf)
        tfidf_matrix = tfidf.fit_transform(df[col])
 
        # Truncated SVD
        # 确保 n_components 不超过特征数或样本数
        effective_n_components = min(n_components, tfidf_matrix.shape[1] - 1, df.shape[0] - 1)
        if effective_n_components < 1:
             print(f"  警告: 列 {col} 的有效 SVD 组件数 ({effective_n_components}) 小于 1，无法进行 SVD。跳过此列的 SVD。")
             # 可以选择跳过 SVD 或添加 0 特征
             # 这里我们跳过 SVD
             continue
 
        print(f"  进行 Truncated SVD (n_components={effective_n_components})...")
        svd = TruncatedSVD(n_components=effective_n_components, random_state=42)
 
        try:
             svd_matrix = svd.fit_transform(tfidf_matrix)
             all_text_features.append(svd_matrix)
             # 生成特征名
             current_feature_names = [f'{col}_svd_{i}' for i in range(effective_n_components)]
             final_text_feature_names.extend(current_feature_names)
             print(f"  完成 {col} 的 SVD 特征提取，维度: {svd_matrix.shape}")
        except ValueError as e:
             print(f"  错误: 对列 {col} 进行 SVD 时出错: {e}。跳过此列的 SVD。")
             # 可以选择跳过或添加 0 特征
             continue
 
 
    if not all_text_features:
         print("没有成功提取任何文本特征。")
         return pd.DataFrame(index=df.index), [] # 返回空的 DataFrame 和列表
 
    # 合并所有文本特征
    print("合并所有提取的文本特征...")
    final_text_features_matrix = np.hstack(all_text_features)
    print(f"最终文本特征矩阵维度: {final_text_features_matrix.shape}")
 
    # 创建 DataFrame
    text_features_df = pd.DataFrame(final_text_features_matrix, columns=final_text_feature_names, index=df.index)
    print("文本特征处理完成。")
    return text_features_df, final_text_feature_names


def process_text_features(df, text_columns=["title", "content", "cover_ocr_content", "video_content"], n_components=50, max_features_tfidf=5000):
    """使用 TF-IDF 和 TruncatedSVD 处理文本列"""
    all_text_features = []
    final_text_feature_names = []
    for col in text_columns:
        if col not in df.columns:
            continue
        df[col] = df[col].fillna("")
        tfidf = TfidfVectorizer(max_features=max_features_tfidf)
        tfidf_matrix = tfidf.fit_transform(df[col])
        effective_n_components = min(n_components, tfidf_matrix.shape[1] - 1, df.shape[0] - 1)
        if effective_n_components < 1:
            continue
        svd = TruncatedSVD(n_components=effective_n_components, random_state=42)
        try:
            svd_matrix = svd.fit_transform(tfidf_matrix)
            all_text_features.append(svd_matrix)
            current_feature_names = [f'{col}_svd_{i}' for i in range(effective_n_components)]
            final_text_feature_names.extend(current_feature_names)
        except Exception:
            continue
    if not all_text_features:
        return pd.DataFrame(index=df.index), []
    final_text_features_matrix = np.hstack(all_text_features)
    text_features_df = pd.DataFrame(final_text_features_matrix, columns=final_text_feature_names, index=df.index)
    return text_features_df, final_text_feature_names


if __name__ == "__main__":
    # 指定要处理的文件路径
    file_to_process = r"D:\study\codes\elm\Elm\Dataset\B\B.txt"
    
    # 读取数据
    print(f"开始读取文件: {file_to_process}")
    try:
        df_full = pd.read_csv(file_to_process, sep="\t")
        print(f"文件读取成功，数据形状: {df_full.shape}")
    except FileNotFoundError:
        print(f"错误: 文件 {file_to_process} 未找到。请检查路径是否正确。")
        exit()
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        exit()

    # 自动检测可用的文本列
    default_text_columns = ['title', 'content', 'cover_ocr_content', 'video_content']
    available_text_columns = [col for col in default_text_columns if col in df_full.columns]
    
    if not available_text_columns:
        print(f"警告: 在文件 {file_to_process} 中未找到默认的文本列 ({', '.join(default_text_columns)})。")
        print(f"文件中的列为: {df_full.columns.tolist()}")
        # 尝试使用文件中的第一列作为文本列（如果存在且为文本类型）
        if not df_full.empty and len(df_full.columns) > 0 and pd.api.types.is_string_dtype(df_full.iloc[:, 0]):
             first_col_name = df_full.columns[0]
             print(f"将尝试使用第一列 '{first_col_name}' 作为文本列进行处理。")
             available_text_columns = [first_col_name]
        else:
             print("无法自动确定可处理的文本列，程序退出。请确保文件中至少有一列文本数据，或在调用时明确指定 text_columns 参数。")
             exit()

    print(f"将使用以下文本列进行处理: {available_text_columns}")
    
    # 设定 SVD 的 n_components 和 TF-IDF 的 max_features
    n_svd_components = 50 
    max_tfidf_features = 5000

    df_text_features, text_feature_names = process_text_features(
        df_full.copy(), # 使用 .copy() 以避免 SettingWithCopyWarning
        text_columns=available_text_columns,
        n_components=n_svd_components,
        max_features_tfidf=max_tfidf_features
    )

    print("\n--- 生成的文本特征 ---")
    if not df_text_features.empty:
        print(f"特征DataFrame形状: {df_text_features.shape}")
        print("特征列名:")
        # 打印部分特征名以避免过多输出
        for i, name in enumerate(text_feature_names):
            if i < 10 or i >= len(text_feature_names) - 5: # 打印前10个和后5个
                 print(f"  - {name}")
            elif i == 10:
                 print(f"  ... (还有 {len(text_feature_names) - 15} 个特征名未显示) ...")
        print("\n特征DataFrame (前5行):")
        print(df_text_features.head())
    else:
        print("未能生成文本特征。")
