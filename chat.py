import torch
from model import GPT, GPTConfig


out_dir = 'out-shakespeare-char'
checkpoint = torch.load(f'{out_dir}/ckpt.pt', map_location='cpu')
model = GPT(checkpoint['model_args'])
model.load_state_dict(checkpoint['model'])
model.eval()

print("NanoGPT Chatbot ready! Type your prompt below (type 'exit' to quit).\n")

while True:
    prompt = input("You: ")
    if prompt.lower() == 'exit':
        break
    
