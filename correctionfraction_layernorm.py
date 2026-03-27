from model_final import GPT, GPTConfig
import torch
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

block_size = 32
vocab_size = 128
device = 'cuda'
wlnorm = 'without'

nathan_checkpoints = {'without': 'sortgpt_k32_methfixed_mlp1_L2_N128_E64_pos0_fln0_wd0p0_seed1337__final.pt',
                      'with': 'sortgpt_k32_methfixed_mlp1_L2_N128_E64_pos0_fln1_wd0p0_seed1337__final.pt'}
checkpoint = torch.load(f'./saved_models/nathanmodels/{nathan_checkpoints[wlnorm]}', map_location=device, weights_only=False)
config = GPTConfig(**checkpoint['model_config'])
model = GPT(config)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device=device)
model.eval()


def get_batch(batch_size=1):
    def cat_sorted_tensor(x):
        vals, _ = torch.sort(x)
        return torch.cat((x, torch.tensor([vocab_size]), vals), dim=0)
    x = torch.stack([cat_sorted_tensor(torch.randperm(vocab_size)[:block_size]) for _ in range(batch_size)])
    return x


num_tries = 100
seq_len = 2 * block_size + 1

one_wrong_perloc = np.zeros(block_size)
both_wrong_perloc = np.zeros(block_size)
one_wrong_clogit_perloc = np.zeros(block_size)
both_wrong_clogit_perloc = np.zeros(block_size)

for trial in range(num_tries):
    idx = get_batch(1).to(device=device)
    with torch.no_grad():
        logits, _ = model(idx, block_size=block_size, return_full_logits=True)
    is_correct = (torch.argmax(logits[0, block_size:2*block_size, :], dim=1) == idx[0, block_size+1:])

    for j in range(block_size, 2 * block_size):
        next_num = idx[0, j + 1].item()
        loc = j - block_size

        num_correct_scores = 0
        for layer in range(2):
            attn = model.transformer.h[layer].attn.stored_attn
            max_score = float('-inf')
            max_num = -1
            for k in range(seq_len):
                s = attn[j, k].item()
                if s > max_score:
                    max_score = s
                    max_num = idx[0, k].item()
            if max_num == next_num:
                num_correct_scores += 1

        if num_correct_scores == 1:
            one_wrong_perloc[loc] += 1
            if is_correct[loc]:
                one_wrong_clogit_perloc[loc] += 1
        elif num_correct_scores == 0:
            both_wrong_perloc[loc] += 1
            if is_correct[loc]:
                both_wrong_clogit_perloc[loc] += 1

one_wrong_perloc /= num_tries
both_wrong_perloc /= num_tries
one_wrong_clogit_perloc /= num_tries
both_wrong_clogit_perloc /= num_tries

data_dir = f'plots_{wlnorm}layernorm'
os.makedirs(data_dir, exist_ok=True)
np.savez(f'{data_dir}/correction_fraction_data.npz',
         one_wrong_perloc=one_wrong_perloc,
         both_wrong_perloc=both_wrong_perloc,
         one_wrong_clogit_perloc=one_wrong_clogit_perloc,
         both_wrong_clogit_perloc=both_wrong_clogit_perloc)
print(f'Data saved to {data_dir}/correction_fraction_data.npz')

# Plot
fig, ax = plt.subplots(figsize=(7, 4))
locs = np.arange(block_size)
bw = 0.38

ax.bar(locs - bw/2, one_wrong_perloc, bw,
       color='#2c7bb6', alpha=0.85,
       label='One attention wrong')
ax.bar(locs + bw/2, both_wrong_perloc, bw,
       color='#d7191c', alpha=0.85,
       label='Both attentions wrong')

ax.set_xlabel('Position in output sequence', fontsize=12)
ax.set_ylabel('Fraction', fontsize=12)
ax.set_title(f'Incorrect attention scores per position — {wlnorm} LayerNorm',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=10, framealpha=0.9, loc='upper right')
ax.grid(True, axis='y', alpha=0.2, linestyle=':')
ax.tick_params(labelsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout()

os.makedirs('plots', exist_ok=True)
output_path = f'plots/correction_fraction_{wlnorm}layernorm.png'
fig.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()
print(f'Plot saved to {output_path}')
