from model_tbyt_3 import GPT, GPTConfig
import torch
import os
#block_size = 8
#vocab_size = 128
block_size = 32
vocab_size = 128
instance_num = 1
device = 'cuda'
config = GPTConfig(block_size=block_size, vocab_size=vocab_size)
model = GPT(config)
#model_state_dict = torch.load(os.path.join(os.getcwd(), f'saved_models/tbyt_1head_2_itr:{itr_num}_checkpoint_old.pt'), map_location=device)['model']
model_state_dict = torch.load(os.path.join(os.getcwd(), f'saved_models/dec28_tbyt_without-pos-embedding_n_embd:64_1head_layers:2_vocab_size:128_itr:60000_checkpoint.pt'), map_location=device)['model']
#model_state_dict = torch.load('./saved_models/tbyt_b64_v2048_embd16_1head_2_itr:20000_checkpoint.pt', map_location=device)['model']
model.load_state_dict(model_state_dict)
model.to(device=device)

def get_batch(batch_size=batch_size):
   def cat_sorted_tensor(x):
      vals, _ = torch.sort(x)
      return torch.cat((x, torch.tensor([vocab_size]), vals), dim=0)
   #x = torch.stack([cat_sorted_tensor(torch.randperm(vocab_size)[:block_size]) for _ in range(batch_size)])
   x = torch.stack([cat_sorted_tensor(torch.randperm(vocab_size)[:32]) for _ in range(batch_size)])
   return x

threshold = 0.05
for i in range(instance_num):
   idx = get_batch()
   idx = idx.to(device=device)
   logits, loss = model(idx)
   all_candidates = []
   candidates = [[] for j in range(block_size, 2*block_size + 1)]
   for j in range(block_size, 2*block_size + 1):
        for k in range(0, 2*block_size + 1):
            if model.transformer.h[0].c_attn.attn[j,k] >= threshold:
                candidates[j-block_size].append((k,idx[k]))
                all_candidates.append((k,idx[k]))
        print(f'candidates for position {j} are {candidates[j-block_size]}\n')    

  