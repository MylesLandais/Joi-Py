import ollama


# Check local model list
ol_list = ollama.list()

print(ol_list)
# TODO: refactor the var list of model names
names = [model["name"] for model in ol_list["models"]]
print(names)
print('unknown_model' in names) # Should eval to false

# Iterate Models list
# for m in ol_list['models']:
#    print(m['name'])

# Prompt + Chat
msg = [{"role":"user","content":"What is a large language model"}]
stream = ollama.chat(model="gemma:7b", messages=msg)
print(stream)
# Multi-Modal Ollama
img = "joi.jpg"

# Check if model exists (ollama generate will error if not found)
if "llava" not in names:
    ollama.pull('llava')

#Multimodal test
try:
    response = ollama.generate(model="llava", prompt="Please describe the image...", images=[img])
    print(response)
except Exception as E:
    # Breaks if you dont have the model pulled..
    print(E)
