import re
import pickle
import numpy as np
import sys

"""

"""
##RESULT_FILE = sys.argv[1]
RESULT_FILE = "/Users/haoming/Desktop/MPI-main/models/gpt-3/path-to-save-p1.pickle"
with open(RESULT_FILE, 'rb+') as f:
    all_results = pickle.load(f)

# 打开.pickle文件
with open('/Users/haoming/Desktop/MPI-main/models/gpt-3/path-to-save-p1.pickle', 'rb') as file:
    # 使用pickle模块加载文件内容
    data = pickle.load(file)
# 打印读取到的内容
print(data)

count = {'A' : 0, 'B' : 0, 'C' : 0, 'D' : 0, 'E': 0,'UNK' : 0}
traits = {
    "O" : [],
    "C" : [],
    "E" : [],
    "A" : [],
    "N" : []
}

SCORES = {
    "A" : 5 , 
    "B" : 4 ,
    "C" : 3 ,
    "D" : 2 ,
    "E" : 1 ,
}

def calc_mean_and_var(result):
    mean = {}  # 用于存放各个特征的均值
    std = {}  # 用于存放各个特征的标准差
    for key, item in result.items():  # 遍历result字典的键值对
        mean[key] = np.mean(np.array(item))  # 计算每个特征的均值并存入mean字典
        std[key] = np.std(np.array(item))  # 计算每个特征的标准差并存入std字典

    return f'''mean:\n {sorted(mean.items(), key=lambda item:item[0])}\n std:\n {sorted(std.items(), key=lambda item:item[0])}'''
    # 返回结果字符串，包含各个特征的均值和标准差，按特征名称排序后输出

for question in all_results:  # 遍历all_results列表中的每个元素
    res = question[2]['text'] + ')'  # 获取问题文本并在末尾添加括号
    choice = re.search(r'[abcdeABCDE][^a-zA-Z]', res, flags=0).group()[0].upper()
    # 使用正则表达式寻找匹配的选项字母，并转换为大写
    count[choice] += 1  # 统计各个选项出现的次数
    label = question[0]['label_ocean']  # 获取问题的标签
 #   label_raw = question[0]['label_raw']  # 获取问题的原始标签
    label_raw = question[0]['explanation']  # 获取问题的原始标签
    key = question[0]['key']  # 获取问题的键值
    score = SCORES[choice]  # 根据选项字母获取对应的分数

    if key == 1:  # 如果键值为1
        traits[label].append(score)  # 将分数添加到对应标签的特征列表中
    else:
        traits[label].append(6 - score)  # 将6减去分数后添加到对应标签的特征列表中

print(calc_mean_and_var(traits))  # 调用calc_mean_and_var函数计算特征的均值和标准差，并打印结果

print(count)  # 打印选项出现的次数统计结果