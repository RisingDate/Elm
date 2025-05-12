import pandas as pd

# 读取TXT文件
file_path = '../../Dataset/A/train.txt'
data = pd.read_csv(file_path, sep='\t')

# 导出到XLSX文件（Excel 2007+格式）
output_file = 'trainA-output.csv'
data.to_csv(output_file, index=False, encoding="utf-8-sig")  # 使用 openpyxl 引擎

print(f"文件已保存为 {output_file}")