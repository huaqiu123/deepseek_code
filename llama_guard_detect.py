import pandas as pd
# 读取 CSV 文件
import torch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1,2,3'
from typing import List, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer




def load_model_and_tokenizer(path):
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    
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

def chat(model, tokenizer):
    df = pd.read_csv("/mnt/data/cyf/qwh/DeepSeek_R1_Distill_Qwen_14B/code/result_answers.csv")
# 遍历每一行，获取 answer 列的内容
    for index, row in df.iterrows():
        answer = row['answer']
        print(f"Answer for question {index}: {answer}")
        history=[({"role": "user", "content": answer})]
        text = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096).to('cuda')
        print('Assistant:', end=' ', flush=True)
        response = generate_response(model, tokenizer, model_inputs.input_ids, model_inputs.attention_mask)
        print()
def main():
    print("hello")
    path = "/mnt/data/cyf/qwh/llama_guard_3_8b/model"
    tokenizer, model = load_model_and_tokenizer(path)
    print('Starting chat.')
    chat(model, tokenizer)

if __name__ == "__main__":
    main()
