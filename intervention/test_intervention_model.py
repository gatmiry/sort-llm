from model_tbyt_4 import GPTConfig, GPT, GPTIntervention
import torch
import os
itr_num = 140000
#block_size = 8
#vocab_size = 128
block_size = 32
vocab_size = 128
device = 'cpu'
config = GPTConfig(block_size=block_size, vocab_size=vocab_size)
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


idx = get_batch()
print('idx is ', idx)
logits, loss = model(idx)
print('model output is ', torch.argmax(logits, dim=-1))
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # Use non-interactive backend
plt.plot(model.transformer.h[0].c_attn.attn[34,:].detach().numpy())
plt.title('Original Attention')
plt.savefig('plots_intervented_attention/original_attention.png', dpi=150, bbox_inches='tight')
plt.show()
intervention_model = GPTIntervention(model, idx)
location = 34  
new_model, (unsorted_lb_selected, unsorted_lb_values), (unsorted_ub_selected, unsorted_ub_values), (sorted_actual_indices, sorted_values) = intervention_model.intervent_attention(attention_layer_num=0, 
                                            location=location, 
                                            unsorted_lb=10, 
                                            unsorted_ub=10, 
                                            unsorted_lb_num=1, 
                                            unsorted_ub_num=1, 
                                            unsorted_intensity_inc=1.0, 
                                            sorted_lb=0, 
                                            sorted_num=0, 
                                            sorted_intensity_inc=0.0)

print('location is ', location, ' with value', idx[0, location].item())
print('unsorted_lb_selected is ', unsorted_lb_selected)
print('unsorted_lb_values is ', unsorted_lb_values)
print('unsorted_ub_selected is ', unsorted_ub_selected)
print('unsorted_ub_values is ', unsorted_ub_values)
print('sorted_actual_indices is ', sorted_actual_indices)
print('sorted_values is ', sorted_values)
logits, _ = new_model(idx)
print('logits from new model shape is ', logits.shape)
print('new model output is ', torch.argmax(logits, dim=-1))
plt.plot(new_model.transformer.h[0].c_attn.new_attn[38,:].detach().numpy())
plt.title('Interventioned Attention')
plt.savefig('plots_intervented_attention/interventioned_attention.png', dpi=150, bbox_inches='tight')
plt.show()
