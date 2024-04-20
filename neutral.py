import openai
import pandas as pd
import pickle
import os
import time
openai.api_key="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"
##openai.api_key = os.environ["OPENAI_API_KEY"] ##os.environ 是一个字典，它包含了所有环境变量。您可以像操作普通字典一样操作它。要注意的是，如果您尝试访问不存在的环境变量，os.environ() 会引发 KeyError。
ITEMPATH = "/Users/haoming/Desktop/MPI-main/models/TestGpt.csv"
TEST_TYPE = None
LABEL_TYPE = None

def getItems(filename=ITEMPATH, item_type=None, label_type=LABEL_TYPE):
    data = pd.read_csv(filename)
    if label_type is not None:
        items = data[data["label_ocean"] == label_type]
    else:
        items = data
    return items


template ="""
You are a very friendly and gregarious person who loves to be around others. 
You are assertive and confident in your interactions, and you have a high activity level. 
You are always looking for new and exciting experiences, and you have a cheerful and optimistic outlook on life.
 
Question: Given the description of you: "You {}." What do you think?

Options:
(A). Very Accurate
(B). Moderately Accurate
(C). Neither Accurate Nor Inaccurate
(D). Moderately Inaccurate
(E). Very Inaccurate

Answer: I choose option
"""

import time

dataset = getItems(ITEMPATH, TEST_TYPE)

for temperature in [0]:
    # 针对不同的温度进行循环，此处温度列表只包含一个元素0

    # 批量处理请求
    batch_size = 20  # 每个批次的大小
    result = []  # 存储结果的列表

    # 分批进行请求
    for i in range(0, len(dataset), batch_size):
        # 根据指定的批次大小，循环遍历数据集
        time.sleep(30)  # 每次请求之间延迟30秒，以遵守API的使用限制
        batch = dataset[i : i + batch_size]  # 获取当前批次的数据
        questions = [
            template.format(item["text"].lower()) for _, item in batch.iterrows()
        ]
        # 生成问题列表，根据模板和数据集中的文本进行格式化

        # 发送请求
        responses = openai.Completion.create(
            engine="gpt-3.5-turbo-instruct",
            prompt=questions,
            temperature=temperature,
            max_tokens=400,
            top_p=0.95,
        )
        # 使用OpenAI API发送请求获取响应

        # 处理响应结果
        for j, response in enumerate(responses["choices"]):
            # 遍历每个问题的响应结果
            result.append((batch.iloc[j], questions[j], response))
            # 将当前批次中的问题、响应和对应的数据项添加到结果列表中
            print(response["text"], batch.iloc[j]["label_ocean"])
            # 打印响应文本和对应数据项的标签（用于调试和查看输出）
        #  answer = response.choices[0].text.strip()



    # 保存结果到pickle文件
    with open(f"path-to-save-p1.pickle", "wb+") as f:
        pickle.dump(result, f)


"""
 # 保存结果到txt文件
    with open("path-to-save-batch11.txt", "w+") as f:
        for item in result:
            f.write(f"{item}\n")

"""