"""
Test intervention using load_checkpoint and GPTInterventionNormalized
for models trained with model_tbyt_3_withnormalization.py
"""
import torch
import os
import sys

# Add parent directory to path for checkpoint_utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from checkpoint_utils import load_checkpoint
from gpt_intervention_normalized import GPTInterventionNormalized

# Load model using load_checkpoint
checkpoint_path = os.path.join(os.path.dirname(__file__), '../Grid_training_without_duplicates/Final_N128_K16_L2_H1_E64_r2over1_npos1_mlp1_dup0_testK16_iters60000.pt')
device = 'cuda'
model, config = load_checkpoint(checkpoint_path, device=device)
model.eval()

# Get config values
block_size = config.block_size
vocab_size = config.vocab_size
vocab_n = vocab_size - 1  # Numbers range from 0 to vocab_n-1, separator is vocab_n

print(f"Loaded model with block_size={block_size}, vocab_size={vocab_size}, vocab_n={vocab_n}")

def get_batch(batch_size=1):
    """Generate a batch of sorting examples for the loaded model"""
    def cat_sorted_tensor(x):
        vals, _ = torch.sort(x)
        # Use vocab_n as separator
        return torch.cat((x, torch.tensor([vocab_n]), vals), dim=0)
    x = torch.stack([cat_sorted_tensor(torch.randperm(vocab_n)[:block_size]) for _ in range(batch_size)])
    return x

# Run intervention experiments
num_rounds = 1000
num_tries = 0
num_successes = 0

for round in range(num_rounds):
    idx = get_batch().to(device)
    
    try:
        intervention_model = GPTInterventionNormalized(model, idx)
        
        # Location in the output part (block_size to 2*block_size)
        # For K=16, valid locations are 16 to 31
        location = block_size + 5  # Position 21 for K=16
        
        ### unsorted up and down intervention
        new_model, _ = intervention_model.intervent_attention(
            attention_layer_num=0, 
            location=location, 
            unsorted_lb=5, 
            unsorted_ub=5, 
            unsorted_lb_num=1, 
            unsorted_ub_num=1, 
            unsorted_intensity_inc=0.0, 
            sorted_lb=0, 
            sorted_num=0, 
            sorted_intensity_inc=0.5
        )
        
        new_generated_number, next_number = intervention_model.check_if_still_works()
        num_successes += (new_generated_number == next_number)
        num_tries += 1
        
        intervention_model.revert_attention(0)
        intervention_model.revert_attention(1)
        
    except Exception as e:
        print(f"Exception in round {round}: {e}")
        continue

if num_tries > 0:
    print(f'Number of successes: {num_successes}, Number of tries: {num_tries}, Success rate: {num_successes / num_tries:.4f}')
else:
    print("No successful trials completed")
