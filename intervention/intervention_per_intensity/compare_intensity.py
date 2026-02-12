"""
Compare intervention intensity effects between models with and without layer normalization.
Plots both curves in the same plot.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from model_tbyt_4 import GPTConfig, GPT, GPTIntervention
from checkpoint_utils import load_checkpoint
from gpt_intervention_normalized import GPTInterventionNormalized
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Config
device = 'cuda'
num_rounds = 1000
FN = 200
intensity_values = [-0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]

# =====================
# Model WITHOUT layer normalization
# =====================
print("=" * 50)
print("Testing model WITHOUT layer normalization")
print("=" * 50)

block_size_1 = 32
vocab_size_1 = 128
vocab_n_1 = vocab_size_1 - 1
location_1 = 45

config_1 = GPTConfig(block_size=block_size_1, vocab_size=vocab_size_1)
model_1 = GPT(config_1)
checkpoint_path_1 = os.path.join(os.path.dirname(__file__), '../../saved_models/dec28_tbyt_without-pos-embedding_n_embd:64_1head_layers:2_vocab_size:128_itr:60000_checkpoint.pt')
model_1.load_state_dict(torch.load(checkpoint_path_1, map_location=device)['model'])
model_1.to(device)
model_1.eval()

def get_batch_1():
    x = torch.randperm(vocab_n_1)[:block_size_1]
    vals, _ = torch.sort(x)
    return torch.cat((x, torch.tensor([vocab_n_1]), vals), dim=0).unsqueeze(0)

results_1 = {}
for intensity in intensity_values:
    attempts = []
    for _ in range(num_rounds):
        if len(attempts) >= FN:
            break
        idx = get_batch_1().to(device)
        intervention_model = GPTIntervention(model_1, idx)
        try:
            intervention_model.intervent_attention(
                attention_layer_num=0, location=location_1,
                unsorted_lb=5, unsorted_ub=5, unsorted_lb_num=0, unsorted_ub_num=1,
                unsorted_intensity_inc=intensity, sorted_lb=0, sorted_num=0, sorted_intensity_inc=0.0
            )
            new_generated_number, next_number = intervention_model.check_if_still_works()
            attempts.append(new_generated_number == next_number)
            intervention_model.revert_attention(0)
        except:
            continue
    first_fn = attempts[:FN]
    results_1[intensity] = sum(first_fn) / len(first_fn) if len(first_fn) > 0 else 0
    print(f'Intensity: {intensity:+.2f}, Success rate: {results_1[intensity]:.4f} ({sum(first_fn)}/{len(first_fn)})')

# =====================
# Model WITH layer normalization
# =====================
print("\n" + "=" * 50)
print("Testing model WITH layer normalization")
print("=" * 50)

checkpoint_path_2 = os.path.join(os.path.dirname(__file__), '../../Grid_training_without_duplicates/Final_N128_K16_L2_H1_E32_r4over1_npos1_mlp1_dup0_testK16_iters60000.pt')
model_2, config_2 = load_checkpoint(checkpoint_path_2, device=device)
model_2.eval()

block_size_2 = config_2.block_size
vocab_n_2 = config_2.vocab_size - 1
location_2 = block_size_2 + 5

def get_batch_2():
    x = torch.randperm(vocab_n_2)[:block_size_2]
    vals, _ = torch.sort(x)
    return torch.cat((x, torch.tensor([vocab_n_2]), vals), dim=0).unsqueeze(0)

results_2 = {}
for intensity in intensity_values:
    attempts = []
    for _ in range(num_rounds):
        if len(attempts) >= FN:
            break
        idx = get_batch_2().to(device)
        try:
            intervention_model = GPTInterventionNormalized(model_2, idx)
            intervention_model.intervent_attention(
                attention_layer_num=0, location=location_2,
                unsorted_lb=5, unsorted_ub=5, unsorted_lb_num=0, unsorted_ub_num=1,
                unsorted_intensity_inc=intensity, sorted_lb=0, sorted_num=0, sorted_intensity_inc=0.0
            )
            new_generated_number, next_number = intervention_model.check_if_still_works()
            attempts.append(new_generated_number == next_number)
            intervention_model.revert_attention(0)
        except:
            continue
    first_fn = attempts[:FN]
    results_2[intensity] = sum(first_fn) / len(first_fn) if len(first_fn) > 0 else 0
    print(f'Intensity: {intensity:+.2f}, Success rate: {results_2[intensity]:.4f} ({sum(first_fn)}/{len(first_fn)})')

# =====================
# Plot comparison
# =====================
# Single column width for two-column paper (~3.5 inches)
plt.figure(figsize=(3.5, 2.8))
intensities = list(results_1.keys())

plt.plot(intensities, list(results_1.values()), marker='o', linewidth=1.5, markersize=5, 
         label='without layer norm', color='#1f77b4')
plt.plot(intensities, list(results_2.values()), marker='s', linewidth=1.5, markersize=5, 
         label='with layer norm', color='#ff7f0e')

plt.xlabel('Intervention Intensity', fontsize=9)
plt.ylabel('Success Probability', fontsize=9)
plt.title(f'Robustness to Attention Intervention', fontsize=10)
plt.legend(fontsize=7, loc='lower left')
plt.grid(True, alpha=0.3)
plt.xticks(intensities[::2], fontsize=8)  # Show every other tick to reduce clutter
plt.yticks(fontsize=8)
plt.ylim(0, 1.05)
plt.tight_layout()

output_dir = os.path.dirname(__file__)
plt.savefig(os.path.join(output_dir, 'compare_intensity.png'), dpi=300, bbox_inches='tight')
print(f'\nPlot saved to {os.path.join(output_dir, "compare_intensity.png")}')
