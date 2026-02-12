import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from model_tbyt_4 import GPTConfig, GPT, GPTIntervention
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Config
block_size = 32
vocab_size = 128
device = 'cpu'
num_rounds = 1000  # Total rounds to attempt
FN = 200  # First N successful attempts to use for comparison
location = 45

# Load model
config = GPTConfig(block_size=block_size, vocab_size=vocab_size)
model = GPT(config)
checkpoint_path = os.path.join(os.path.dirname(__file__), '../../saved_models/dec28_tbyt_without-pos-embedding_n_embd:64_1head_layers:2_vocab_size:128_itr:60000_checkpoint.pt')
model.load_state_dict(torch.load(checkpoint_path, map_location=device)['model'])
model.to(device)
model.eval()

# Batch generation
vocab_n = vocab_size - 1
def get_batch():
    x = torch.randperm(vocab_n)[:block_size]
    vals, _ = torch.sort(x)
    return torch.cat((x, torch.tensor([vocab_n]), vals), dim=0).unsqueeze(0)

# Test different intensity values
intensity_values = [-0.5, -0.25, 0.0, 0.25, 0.5]
all_results = {}  # Store all attempts per intensity

for intensity in intensity_values:
    attempts = []  # List of (success: bool) for each valid attempt
    
    for _ in range(num_rounds):
        if len(attempts) >= FN:
            break
            
        idx = get_batch().to(device)
        intervention_model = GPTIntervention(model, idx)
        
        try:
            new_model, _ = intervention_model.intervent_attention(
                attention_layer_num=0,
                location=location,
                unsorted_lb=5,
                unsorted_ub=5,
                unsorted_lb_num=0,
                unsorted_ub_num=1,
                unsorted_intensity_inc=intensity,
                sorted_lb=0,
                sorted_num=0,
                sorted_intensity_inc=0.0
            )
            
            new_generated_number, next_number = intervention_model.check_if_still_works()
            attempts.append(new_generated_number == next_number)
            intervention_model.revert_attention(0)
        except:
            continue
    
    all_results[intensity] = attempts
    print(f'Intensity: {intensity:+.2f}, Valid attempts: {len(attempts)}')

# Compute success rate using first FN attempts for each intensity
results = {}
for intensity, attempts in all_results.items():
    first_fn = attempts[:FN]
    success_rate = sum(first_fn) / len(first_fn) if len(first_fn) > 0 else 0
    results[intensity] = success_rate
    print(f'Intensity: {intensity:+.2f}, Success rate (first {FN}): {success_rate:.4f} ({sum(first_fn)}/{len(first_fn)})')

# Plot
plt.figure(figsize=(8, 6))
intensities = list(results.keys())
success_rates = list(results.values())
plt.plot(intensities, success_rates, marker='o', linewidth=2, markersize=10, color='#1f77b4')
plt.xlabel('Intervention Intensity')
plt.ylabel('Success Probability')
plt.title(f'Success Probability vs Intervention Intensity ({FN} attempts)')
plt.grid(True, alpha=0.3)
plt.xticks(intensities)
plt.ylim(0, 1.05)

# Save plot
output_dir = os.path.dirname(__file__)
plt.savefig(os.path.join(output_dir, 'success_vs_intensity.png'), dpi=150, bbox_inches='tight')
print(f'\nPlot saved to {os.path.join(output_dir, "success_vs_intensity.png")}')
