import pandas as pd

from dataProcess import data_process

params = {
    'res1_path': '../../Me',
    'res2_path': '',
    'test_data_path': '../../../Dataset/A/test_data.txt'
}
if __name__ == '__main__':
    res_path1 = '../../MachineLearning/results/catboost_output.txt'
    res_path2 = '../../DeepLearning/results/B/output-250516-1-tf-all-data.txt'
    save_path_txt = '../results/B/output-average.txt'
    save_path_csv = '../results/B/output-average.csv'
    df1 = pd.read_csv(res_path1, sep='\t')
    df2 = pd.read_csv(res_path2, sep='\t')

    if not (df1['id'].equals(df2['id'])):
        raise ValueError("❌ 两个结果文件的 id 列不一致，无法对齐融合。")
    df_ensemble = df1.copy()
    df_ensemble['interaction_cnt'] = (df1['interaction_cnt'] * 0.2 + df2['interaction_cnt'] * 0.8).round().astype(int)

    df_ensemble.to_csv(save_path_txt, sep='\t', index=False, header=True)
    df_ensemble.to_csv(save_path_csv, header=True, index=False)
    print(f"✅ 融合结果已保存至: {save_path_txt}")
