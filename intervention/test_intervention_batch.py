from model_tbyt_4 import GPTConfig, GPT, GPTIntervention
from test_intervention_model import get_batch
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


num_rounds = 1000
num_tries = 0
num_successes = 0
for round in range(num_rounds):
    idx = get_batch().to(device)
    intervention_model = GPTIntervention(model, idx)
    #location = torch.randint(block_size + 1, block_size * 2 + 1, (1,)).item()
    location = 45
    ### unsorted up and dnwon intervention
    try:
        new_model, _ = intervention_model.intervent_attention(attention_layer_num=0, 
                                            location=location, 
                                                unsorted_lb=5, 
                                                unsorted_ub=5, 
                                                unsorted_lb_num=0, 
                                                unsorted_ub_num=1, 
                                                unsorted_intensity_inc=0.5, 
                                                sorted_lb=0, 
                                                sorted_num=0, 
                                                sorted_intensity_inc=0.5)
        #new_model, _ = intervention_model.intervent_attention(attention_layer_num=1, 
        #                                        location=location, 
        #                                        unsorted_lb=10, 
        #                                        unsorted_ub=10, 
        #                                        unsorted_lb_num=1, 
        #                                        unsorted_ub_num=1, 
        #                                        unsorted_intensity_inc=2.0, 
        #                                        sorted_lb=0, 
        #                                        sorted_num=0, 
        #                                        sorted_intensity_inc=0.5)
        
        new_generated_number, next_number = intervention_model.check_if_still_works()
        num_successes += (new_generated_number == next_number)
        num_tries += 1
        intervention_model.revert_attention(0)
        intervention_model.revert_attention(1)
    except Exception as e:
        print(f"Exception in round {round}: {e}")
        continue
    #num_tries += 1
print(f'Number of successes: {num_successes}, Number of tries: {num_tries}, Success rate: {num_successes / num_tries}')