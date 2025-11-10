import torch
from nanoGPT import GPT, GPTConfig
from transformers import AutoTokenizer

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(device)
config = GPTConfig()

path = ''
model = GPT(config).to(device)
checkpoint = torch.load(path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2-0.5B', trust_remote_code=True)
prompt = '数学是'
input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long).to(device)

with torch.no_grad():
    output_ids = model.generate(input_ids, max_new_tokens=100)
generated_text = tokenizer.decode(output_ids[0].tolist())
print(generated_text)