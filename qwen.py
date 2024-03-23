# based on qwen-vl-chat

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from huggingface_hub import snapshot_download

# Define Model Directory
chkpt = "4bit/Qwen-VL-Chat-Int4"
model_dir = snapshot_download(chkpt)
# Load the model
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", trust_remote_code=True).eval()
model.generation_config = GenerationConfig.from_pretrained(model_dir, trust_remote_code=True)

# TODO: Test no img prompt~

# Inference/Chat
query = tokenizer.from_list_format([
    {'image':'https://cdn.discordapp.com/attachments/1212843143574192248/1215879742239281271/140945409.webp?ex=65fe5b40&is=65ebe640&hm=782b718f7bbefc622a237d9d37f580cd6cbfbe8c4775d63bf2119e758aca84b4&'},
    {'text': 'Describe the Image?'},
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
    {'text': 'Write comma seperated tags describing the img with specific keywords'}
])
response_3, history = model.chat(tokenizer, query=query_3,history=history)
print(response_3)
print(history)
