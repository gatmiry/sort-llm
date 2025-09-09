from model_tbyt import GPT, GPTConfig
import torch
## make sure the batch_size and vocab_size are compatible with file train_tbyt.py
block_size = 32
vocab_size = 2048
device = 'cpu'
model = GPT(GPTConfig(block_size=block_size, vocab_size=vocab_size))
model_state_dict = torch.load('./saved_models/tbyt_1head_2_itr:60000_checkpoint.pt', map_location=device)['model']
#model_state_dict = torch.load('./saved_models/tbyt_b64_v2048_embd16_1head_2_itr:20000_checkpoint.pt')['model']
model.load_state_dict(model_state_dict)
model.to(device)
model.eval()
idx = torch.randint(vocab_size, (32,))
#print('type vocab size ', torch.tensor([vocab_size]).dtype)
idx = torch.cat((idx, torch.tensor([vocab_size])), dim=0).unsqueeze(0).to(device)
## add the first element of the sorted sequence

#minar, _ = torch.min(idx, dim=-1)
#idx = torch.cat((idx, minar.unsqueeze(-1)), dim=-1)
print('sec is ', idx)
sorted_idx = model.generate(idx, topk=1, sampling_length=64)
vals, indices = torch.sort(idx)
print('actual sorted is ', vals)
print('sorted is ', sorted_idx[:,:])

###
batch_size = 1
def get_batch():
   def cat_sorted_tensor(x):
      #x, _ = torch.sort(x, descending=True)
      #x = torch.tensor([100,100,100,100,1,1,1,1])
      vals, _ = torch.sort(x)
      #vals2, _ = torch.sort(x, descending=True)
      #print('vals are ', vals)
      return torch.cat((x, torch.tensor([vocab_size]), vals), dim=0)
   x = torch.stack([cat_sorted_tensor(torch.randperm(vocab_size)[:block_size]) for _ in range(batch_size)])
   return x
###
loss_batch = get_batch()
logits, loss = model(loss_batch)
print(f'loss batch is {loss_batch}')
print(f'this loss is {loss.item()}')
