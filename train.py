import torch
block_size = 8
batch_size = 1024
vocab_size = 128
import torch
import numpy as np
import os
#from torch.func import vmap

def sample_perm(_):
   return torch.randperm(vocab_size)[:block_size]
print('hi!!!')
def get_batch():
   x = []
   y = []
   
   x = torch.stack([torch.randperm(vocab_size)[:block_size] for _ in range(batch_size)])
   #x = vmap(sample_perm, randomness='different')(torch.arange(batch_size))
   y = torch.sort(x, dim=1).values   
   return x, y

def get_batch_tbyt():
   def cat_sorted_tensor(x):
      vals, indices = torch.sort(x)
      return torch.cat((x, vals), dim=0)
   x = torch.stack([cat_sorted_tensor(torch.randperm(vocab_size)[:block_size]) for _ in range(batch_size)])
   return x

import math
warmup_iters = 100
max_iters = 20000
learning_rate = 1e-3
min_lr = 1e-5
decay_lr = True
def get_lr(itr):
    if itr < warmup_iters:
       return learning_rate * (itr + 1) / (warmup_iters + 1)
    if itr > max_iters:
       return min_lr
    assert warmup_iters <= itr <= max_iters, f'itr is out of bound'
    ratio = (itr - warmup_iters) / (max_iters - warmup_iters)
    ratio = 0.5 * (1.0 + math.cos(math.pi * ratio))
    lr = min_lr + ratio * (learning_rate - min_lr)
    return lr


def create_optimizer(model, weight_decay, learning_rate, device):
   params = [p for p in model.parameters() if p.requires_grad]
   decay_params = [p for p in params if p.dim() > 1]
   nondecay_params = [p for p in params if p.dim() <= 1]
   optim_groups = [
      {'params': decay_params, 'weight_decay': weight_decay},
      {'params': nondecay_params, 'weight_decay': 0.0}
   ]
   num_decay_params = sum(p.numel() for p in decay_params)
   num_nondecay_params = sum(p.numel() for p in nondecay_params)
   print(f'num decay params: {num_decay_params} num nondecay params: {num_nondecay_params}')
   import inspect
   #fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
   #use_fused = fused_available and 'cuda' in device
   #print(f'using fused Adam: {use_fused}')
   optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8)#, fused=False)
   return optimizer




from model import GPT, GPTConfig
print('im here!')
mymodel = GPT(GPTConfig(block_size=block_size, vocab_size=vocab_size))
device = 'cpu'
if torch.cuda.is_available():
   device = 'cuda'
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
   device = 'mps'
print(f'using device: {device}')
mymodel.to(device)

optimizer = create_optimizer(mymodel, weight_decay=0.1, learning_rate=6e-4, device=device)

max_iter = 25000
for itr in range(max_iter):
   #print(f'itr: {itr}')
   optimizer.zero_grad()
   x, y = get_batch()
   x,y = x.to(device), y.to(device)
   #print('device: ', device)
   #print(f'model device: {next(mymodel.parameters()).device} x device: {x.device}')
   logits, loss = mymodel(x, y)
   #print('I computed the loss')
   loss.backward()
   #print('did backward')
   lr = get_lr(itr)
   for param_group in optimizer.param_groups:
      param_group['lr'] = lr
   optimizer.step()
   #print('optimizer took step')
   ## computing test loss
   mymodel.eval()
   x, y = get_batch()
   x, y = x.to(device), y.to(device)
   _, test_loss = mymodel(x, y)
   mymodel.train()
   if itr % 100 == 0:
      print(f'itr: {itr} train loss: {loss.item()} test loss: {test_loss.item()}')
   if itr % 5000 == 0:
      checkpoint = {
         'model': mymodel.state_dict(),
         'optimizer': optimizer.state_dict()
      }
      import os
      torch.save(checkpoint, os.path.join(os.getcwd(), f'./saved_models/itr_{itr}_checkpoint.pt'))
