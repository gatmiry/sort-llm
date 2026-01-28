import torch
block_size = 32
batch_size = 4096
vocab_size = 128 #1024
import torch
import numpy as np
import os
from datetime import datetime
#from torch.func import vmap


def get_batch():
   def cat_sorted_tensor(x):
      vals, _ = torch.sort(x)
      #print('vals are ', vals)
      return torch.cat((x, torch.tensor([vocab_size]), vals), dim=0)
   x = torch.stack([cat_sorted_tensor(torch.randperm(vocab_size)[:block_size]) for _ in range(batch_size)])
   return x

import math
warmup_iters = 100
max_iters = 20000
max_iter = 120000
learning_rate = 1e-4
min_lr = 1e-6
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




from model_tbyt_train import GPT, GPTConfig
print('im here!')
myconfig = GPTConfig(block_size=block_size, vocab_size=vocab_size)
mymodel = GPT(myconfig)
device = 'cpu'
if torch.cuda.is_available():
   device = 'cuda'
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
   device = 'mps'
print(f'using device: {device}')
mymodel.to(device)

# Create subfolder for checkpoints with date/time and vocab_size
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
checkpoint_dir = os.path.join('saved_models', f'{timestamp}_vocab{vocab_size}')
os.makedirs(checkpoint_dir, exist_ok=True)
print(f'Checkpoints will be saved to: {checkpoint_dir}')

optimizer = create_optimizer(mymodel, weight_decay=0.1, learning_rate=6e-4, device=device)
import wandb
import time
wandb.init(project='sort-llm', name=f'jan27-with-pos-embedding-1head-max-iter-120000-time-{time.time()}')
for itr in range(max_iter):
   #print(f'itr: {itr}')
   optimizer.zero_grad()
   x = get_batch()
   x = x.to(device)
   #print('device: ', device)
   #print(f'model device: {next(mymodel.parameters()).device} x device: {x.device}')
   logits, loss = mymodel(x)
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
   x = get_batch()
   x = x.to(device)
   if itr % 100 == 0:
      logits, test_loss = mymodel(x, flag=False)
   else:
      logits, test_loss = mymodel(x)
   mymodel.train()
   if itr % 100 == 0:
      print(f'x: {x[0]}')
      vals, indices = torch.topk(logits[0, 8:], 3, -1)
      #print(f'indices: {indices}')
      print(f'itr: {itr} train loss: {loss.item()} test loss: {test_loss.item()}')
      wandb.log({
         'train_loss': loss.item(),
         'test_loss': test_loss.item(),
         'x': x[0]
      })
   if itr % 20000 == 0:
      checkpoint = {
         'model': mymodel.state_dict(),
         'optimizer': optimizer.state_dict()
      }
      checkpoint_filename = f'n_embd:{myconfig.n_embd}_head:{mymodel.config.n_heads}_layers:{mymodel.config.n_layers}_vocab_size:{vocab_size}_itr:{itr}_checkpoint.pt'
      checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)
      torch.save(checkpoint, checkpoint_path)
      print(f'Saved checkpoint to: {checkpoint_path}')
