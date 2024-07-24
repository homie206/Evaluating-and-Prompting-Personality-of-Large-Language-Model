import pandas as pd
import re

# 读取 CSV 文件
df = pd.read_csv('/Users/haoming/Downloads/mbti_1.csv')

# 创建一个新的空列表,用于存储拆分后的结果
new_rows = []

# 遍历 DataFrame 的每一行
for index, row in df.iterrows():
    # 将 'posts' 列的数据按 '|||' 分割成列表
    post_list = row['posts'].split('|||')

    # 对于每个拆分后的子字符串
    for post in post_list:
        # 创建一个新的字典,保存 'type' 列的原始值和拆分后的 'posts' 列值
        new_row = {
            'type': row['type'],
            'posts': post.strip()
        }
        # 将新字典添加到 new_rows 列表中
        new_rows.append(new_row)

# 创建一个新的 DataFrame,并将 new_rows 列表中的数据添加进去
new_df = pd.DataFrame(new_rows)

# 删除包含链接、只包含数字与符号、只包含符号的列,以及 'posts' 列为空的行
new_df = new_df[~new_df['posts'].str.contains('http')]
new_df = new_df[~new_df['posts'].str.contains(r'^[\d\W]+$')]
new_df = new_df[~new_df['posts'].str.contains(r'^[\W]+$')]
new_df = new_df[new_df['posts'].str.len() > 0]

# 将新的 DataFrame 存储到一个新的 CSV 文件中
new_df.to_csv('/Users/haoming/Downloads/mbti_3.csv', index=False)