import pandas as pd
import torch
import os
import csv
from tqdm import tqdm
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from typing import List, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer
def load_model_and_tokenizer(path):
    tokenizer = AutoTokenizer.from_pretrained(path,
		torch_dtype=torch.float16,trust_remote_code=True)
    # 加载模型到多个 GPU 上
    model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True,device_map="auto")
    model.eval()
    return tokenizer, model

def generate_response(model, tokenizer, input_ids, attention_mask, max_new_tokens=4096):
	generated_ids = input_ids
	with torch.inference_mode():
			for _ in range(max_new_tokens):
					outputs = model(input_ids=generated_ids, attention_mask=attention_mask)
					next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(-1)
					generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
					attention_mask = torch.cat([attention_mask, torch.ones_like(next_token_id)], dim=-1)
					new_token = tokenizer.decode(next_token_id.squeeze(), skip_special_tokens=True)
					print(new_token, end='', flush=True)
					if next_token_id.item() == tokenizer.eos_token_id:
							break
	return tokenizer.decode(generated_ids[0][input_ids.shape[-1]:], skip_special_tokens=True)

def produce_result(model, tokenizer, data):
	number=6
# 使用 tqdm 包装 data.iterrows() 来显示进度条
	for index, example in tqdm(data.iloc[6:].iterrows(), total=len(data), desc="Processing", unit="example"):

		user_input = example["prompt"]
		history = [{"role": "user", "content": user_input}]
		
		# 构建生成文本
		text = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
		# 将文本转化为模型输入
		model_inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096).to('cuda')
		# 生成回答
		response = generate_response(model, tokenizer, model_inputs.input_ids, model_inputs.attention_mask, max_new_tokens=4096)
		# 将问题、答案和标签存储在字典中
		new = {
				"question": example["prompt"],
				"answer": response,
				"label": example["label"]
		}
		print(example["label"])
		result_file_path=f'/mnt/data/cyf/qwh/DeepSeek_R1_Distill_Qwen_7B/result/result_answer_{number}.csv'
		with open(result_file_path, mode="w", newline="", encoding="utf-8") as file:
				writer = csv.DictWriter(file, fieldnames=new.keys())
				# 写入表头
				writer.writeheader()
				# 写入数据
				writer.writerow(new)
		number=number+1
	return 0

def main():
    print("hello")
    path = "/mnt/data/cyf/qwh/DeepSeek_R1_Distill_Qwen_7B/model"
    tokenizer, model = load_model_and_tokenizer(path)
    print('Starting produce result.')
    file_path="/mnt/data/cyf/qwh/dataset/xstest/data/train-00000-of-00001.parquet"
    data=prepare_data(file_path)
    result_answer=produce_result(model, tokenizer,data)

def prepare_data(file_path):
    df = pd.read_parquet(file_path)
    return df

if __name__ == "__main__":
    main()
