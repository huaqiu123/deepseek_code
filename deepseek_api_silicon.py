import pandas as pd 
from tqdm import tqdm
import csv
def prepare_data(file_path):
    df = pd.read_parquet(file_path)
    return df

import requests

url = "https://api.siliconflow.cn/v1/chat/completions"
headers = {
    "Authorization": "",
    "Content-Type": "application/json"
}
def get_deepseek_response(question):
	payload = {
			"model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
			"messages": [
					{"role": "user", "content":question}
			],
			"stream": False,
			"max_tokens": 1024,
			"temperature": 0.7,
			"top_p": 0.7,
			"top_k": 50,
			"frequency_penalty": 0.5,
			"n": 1,
			"response_format": {"type": "text"}
	}

	# 发送请求
	response = requests.post(url, json=payload, headers=headers)

	response_data =response.json()
	# 提取模型回答和推理过程
	for choice in response_data.get("choices", []):
		message = choice.get("message", {})
		content = message.get("content", "")
		reasoning_content = message.get("reasoning_content", "")

		# 输出模型的最终回答和推理过程
		print(f"Final Answer: {content}")
		if reasoning_content:
				print(f"Reasoning Chain: {reasoning_content}")
	final="<think>"+reasoning_content+"<think>"+content
	return final


file_path="/mnt/data/qiuweihua/xstest/data/train-00000-of-00001.parquet"
data=prepare_data(file_path)
number=0
for index, example in tqdm(data.iloc[0:50].iterrows(), total=len(data), desc="Processing", unit="example"):
	user_input = example["prompt"]
	response=get_deepseek_response(user_input)
	new = {
			"question": example["prompt"],
			"answer": response,
			"label": example["label"]
	}
	print(example["label"])
	result_file_path=f'/mnt/data/qiuweihua/DeepSeek_R1_Distill_Llama_8B/result/result_answer_{number}.csv'
	with open(result_file_path, mode="w", newline="", encoding="utf-8") as file:
			writer = csv.DictWriter(file, fieldnames=new.keys())
			# 写入表头
			writer.writeheader()
			# 写入数据
			writer.writerow(new)
	number=number+1
	for i in range(1000):
		i=i+1