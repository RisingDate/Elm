import os
from openai import OpenAI
import pandas as pd

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


if __name__ == "__main__":
    test_file = "../../Dataset/B/B.txt"
    df = pd.read_csv(test_file, sep="\t")
    df_sample = df.head(10)
    # 保存临时文件用于测试embedding
    temp_file = "temp_B_head10.txt"
    df_sample.to_csv(temp_file, sep="\t", index=False)
    emb_df = get_text_embeddings(temp_file, column_name="content", dim=1024, batch_size=10)
    print(emb_df)
