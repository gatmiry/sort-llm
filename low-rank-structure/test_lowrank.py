from model_lowrank import GPTConfig, GPT
import torch
import os
itr_num = 140000
#block_size = 8
#vocab_size = 128
block_size = 32
vocab_size = 128
device = 'cpu'
config = GPTConfig(block_size=block_size, vocab_size=vocab_size)
config.n_embd = 64  # Match checkpoint: n_embd:64
config.n_layers = 2  # Match checkpoint: layers:2
model = GPT(config)
#model_state_dict = torch.load(os.path.join(os.getcwd(), f'saved_models/tbyt_1head_2_itr:{itr_num}_checkpoint_old.pt'), map_location=device)['model']
model_state_dict = torch.load(os.path.join(os.getcwd(), f'../saved_models/dec28_tbyt_without-pos-embedding_n_embd:64_1head_layers:2_vocab_size:128_itr:60000_checkpoint.pt'), map_location=device)['model']
#model_state_dict = torch.load('./saved_models/tbyt_b64_v2048_embd16_1head_2_itr:20000_checkpoint.pt', map_location=device)['model']
model.load_state_dict(model_state_dict)
model.to(device=device)
model.eval()
batch_size = 1


def get_batch(changing_num=-1, changing_index=-1, initial_sequence=None, batch_size=batch_size):
   def cat_sorted_tensor(x):
      if initial_sequence is not None:
         x = initial_sequence
      else:
         x = x
         #x, _ = torch.sort(x, descending=True)
      if changing_num != -1:
         if changing_index == -1:
            x[0] = changing_num
         else:
            x[changing_index] = changing_num
      #x = torch.cat((torch.tensor([100]).repeat(16), torch.tensor([1]).repeat(16)))
      #x = torch.tensor([100,100,100,100,1,1,1,1])
      vals, _ = torch.sort(x)
      #vals2, _ = torch.sort(x, descending=True)
      #print('vals are ', vals)
      return torch.cat((x, torch.tensor([vocab_size]), vals), dim=0)
   #x = torch.stack([cat_sorted_tensor(torch.randperm(vocab_size)[:block_size]) for _ in range(batch_size)])
   x = torch.stack([cat_sorted_tensor(torch.randperm(vocab_size)[:32]) for _ in range(batch_size)])
   return x

batch_size = 64
position = 45
idx = get_batch(batch_size=batch_size)
print('idx shape is ', idx.shape)
model.set_svds_once(idx, position)

print('singular values are ', model.transformer.h[0].c_attn.S)
import matplotlib.pyplot as plt
plt.plot(model.transformer.h[0].c_fc.S.detach().numpy())
plt.savefig('figures/singular_values.png')

#### finding the projection
idx = get_batch()
model.find_svd_projection(idx)
plt.plot(model.transformer.h[0].c_fc.projected[:,0].detach().numpy())
plt.savefig('figures/projected.png')
plt.close()


import random
base_index = random.randint(0, block_size - 1)
idx = get_batch(changing_num=0, changing_index=base_index)
print('base index is ', base_index)
print('idx is ', idx)


val = idx[0, position]
print('val is ', val, ' next val is', idx[0, position + 1])
direction_index = 0
direction_vals = []
for num in range(config.vocab_size):
   idx[0, base_index] = num
   model.find_svd_projection(idx)
   direction_vals.append(model.transformer.h[0].c_fc.projected[direction_index,0].detach().numpy())
   
inc = 0.000001
plt.plot(direction_vals)
plt.ylim(ymin=direction_vals[0] - inc, ymax=direction_vals[0] + inc)  # Adjust these values as needed
plt.savefig('figures/direction_vals.png')
plt.show()


#### now testing projection in foreward loop
logits, _ = model(idx)
print('output without projection is ', torch.argmax(logits, dim=-1))
model.hook_attention(attention_layer=0, proj_dimension=10)
logits, loss = model(idx)
print('output is ', torch.argmax(logits, dim=-1))
#print('module names are ', model._modules.keys())
