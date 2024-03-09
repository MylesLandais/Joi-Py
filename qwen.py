# based on qwen-vl-chat

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig



# Load the model
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="auto", trust_remote_code=True).eval()
model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)

# Inference/Chat
query = tokenizer.from_list_format([
    {'image': 'https://www.google.com/logos/doodles/2024/international-womens-day-2024-6753651837110196-l.webp'},
    {'text': 'What does the image say?'},
])
response, history = model.chat(tokenizer, query=query, history=None)
print(response)
query_2 = tokenizer.from_list_format([
    {'text': 'What is the first letter?'}
])
response_2, history = model.chat(tokenizer, query=query_2, history=history)
print(response_2)
print(history)
query_3 = tokenizer.from_list_format([
    {'text': 'What is the url for the domain?'}
])
response_3, history = model.chat(tokenizer, query=query_3,history=history)
print(response_3)
print(history)
