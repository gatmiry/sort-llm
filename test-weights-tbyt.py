from model_tbyt import GPT, GPTConfig
import torch
## make sure the batch_size and vocab_size are compatible with file train_tbyt.py
block_size = 8
vocab_size = 128
model = GPT(GPTConfig(block_size=block_size, vocab_size=vocab_size)).to('cuda')
model_state_dict = torch.load('./saved_models/tbyt_itr:40000_checkpoint.pt')['model']
model.load_state_dict(model_state_dict)
idx = torch.randint(vocab_size, (8,)).unsqueeze(0).to('cuda')
## add the first element of the sorted sequence
minar, _ = torch.min(idx, dim=-1)
idx = torch.cat((idx, minar.unsqueeze(-1)), dim=-1)
print('sec is ', idx)
sorted_idx = model.generate(idx, topk=1, sampling_length=8)
print('sorted is ', sorted_idx[:,:])