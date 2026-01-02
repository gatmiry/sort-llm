from model_tbyt_4 import GPTConfig, GPT, GPTIntervention
import torch
import os
import matplotlib
import matplotlib.pyplot as plt

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
#idx = torch.tensor([[ 77,  65, 105, 107,  13,  26,  89,  62,  72,  57,  69, 115,   2,  59,
#         106,  27,  33,  30,  83,  68,  70,  23,  40,  74,  50,  15,  76, 113,
#         112,  90,  78,  97, 128,   2,  13,  15,  23,  26,  27,  30,  33,  40,
#          50,  57,  59,  62,  65,  68,  69,  70,  72,  74,  76,  77,  78,  83,
#          89,  90,  97, 105, 106, 107, 112, 113, 115]])
print('idx is ', idx)
logits, loss = model(idx)
print('model output is ', torch.argmax(logits, dim=-1))

matplotlib.use('Agg')  # Use non-interactive backend
plt.plot(model.transformer.h[0].c_attn.attn[34,:].detach().numpy())
plt.xlabel("Key / Token index")
plt.ylabel("Attention weight")
plt.title('Original Attention')
plt.savefig('plots_intervented_attention/original_attention.png', dpi=150, bbox_inches='tight')
plt.show()
intervention_model = GPTIntervention(model, idx)
location = 34  
print('model next logit is ', logits[0, location, torch.argmax(logits, dim=-1)[0,location]].item())
new_model, ((unsorted_lb_selected, unsorted_lb_values), (unsorted_ub_selected, unsorted_ub_values), (sorted_actual_indices, sorted_values)) = intervention_model.intervent_attention(attention_layer_num=0, 
                                            location=location, 
                                            unsorted_lb=10, 
                                            unsorted_ub=10, 
                                            unsorted_lb_num=1, 
                                            unsorted_ub_num=1, 
                                            unsorted_intensity_inc=10.0, 
                                            sorted_lb=0, 
                                            sorted_num=0, 
                                            sorted_intensity_inc=0.0)

new_model, _ = intervention_model.intervent_attention(attention_layer_num=1, 
                                            location=location, 
                                            unsorted_lb=10, 
                                            unsorted_ub=10, 
                                            unsorted_lb_num=1, 
                                            unsorted_ub_num=1, 
                                            unsorted_intensity_inc=10.0, 
                                            sorted_lb=0, 
                                            sorted_num=0, 
                                            sorted_intensity_inc=0.0)

new_generated_number, next_number = intervention_model.check_if_still_works()
print('new generated number is ', new_generated_number, ' and next number is ', next_number, '\n')
#intervention_model.revert_attention(0)
#new_generated_number, next_number = intervention_model.check_if_still_works()
#print('new generated number after revert is ', new_generated_number, ' and next number is ', next_number, '\n')

print('location is ', location, ' with value', idx[0, location].item())
print('unsorted_lb_selected is ', unsorted_lb_selected)
print('unsorted_lb_values is ', unsorted_lb_values)
print('unsorted_ub_selected is ', unsorted_ub_selected)
print('unsorted_ub_values is ', unsorted_ub_values)
print('sorted_actual_indices is ', sorted_actual_indices)
print('sorted_values is ', sorted_values)

logits, _ = new_model(idx)
#print('logits from new model shape is ', logits.shape)
output_indices = torch.argmax(logits, dim=-1)
#print('new output index is ', output_indices[0,location])
#print('new model next logit is ', logits[0, location, output_indices[0,location]].item())
plt.plot(new_model.transformer.h[0].c_attn.new_attn[location,:].detach().numpy())
plt.xlabel("Key / Token index")
plt.ylabel("Attention weight")
plt.title('Interventioned Attention')
plt.savefig('plots_intervented_attention/interventioned_attention.png', dpi=150, bbox_inches='tight')
plt.show()
