import torch
itr_checkpoint = 10000
block_size = 8
vocab_size = 128
from model import GPT, GPTConfig
addr = f'./saved_models/bubble_evenn_itr:{itr_checkpoint}.pt'
checkpoint = torch.load(addr, map_location=torch.device('cuda'))
model_state_dict = checkpoint['model']
model = GPT(GPTConfig(block_size=block_size, vocab_size=vocab_size)).to('cuda')
model.load_state_dict(model_state_dict)
model.eval()
idx = torch.randint(vocab_size, (block_size,)).unsqueeze(0).to('cuda')
idx, _ = torch.sort(idx)
print('initial idx ', idx)
idx = model.generate_onepass(idx)
print('secondary idx ', idx)
print('sucess!')