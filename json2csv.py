import json
import csv

# 假设你的JSON文件名为data.json
json_file_path = '/Users/haoming/Downloads/zh_information_sensing.json'
csv_file_path = '/Users/haoming/Downloads/zh_information_sensing.csv'

# 读取JSON文件
with open(json_file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

# 写入CSV文件
with open(csv_file_path, 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['text'])  # 添加列名
    for item in data:
        instruction = item['instruction']
        output = item['output']
        writer.writerow(['human: ' + instruction + '\n bot: ' + output])

print("CSV文件已生成。")