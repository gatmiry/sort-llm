"""
Generate compare_cinclogits plot for Nathan checkpoints using layer 1 attention.
Uses model_final.py. Saves as compare_cinclogits_nathan_layer1.png.
"""
import torch
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model_final import GPT, GPTConfig

device = 'cuda'
block_size = 32
vocab_n = 128
num_tries = 100

nathan_checkpoints = {
    'without': 'saved_models/nathanmodels/sortgpt_k32_methfixed_mlp1_L2_N128_E64_pos0_fln0_wd0p0_seed1337__final.pt',
    'with': 'saved_models/nathanmodels/sortgpt_k32_methfixed_mlp1_L2_N128_E64_pos0_fln1_wd0p0_seed1337__final.pt',
}

def get_batch():
    x = torch.randperm(vocab_n)[:block_size]
    vals, _ = torch.sort(x)
    return torch.cat((x, torch.tensor([vocab_n]), vals), dim=0).unsqueeze(0)

results = {}
for wlnorm in ['without', 'with']:
    checkpoint = torch.load(nathan_checkpoints[wlnorm], map_location=device, weights_only=False)
    config = GPTConfig(**checkpoint['model_config'])
    model = GPT(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    ave_clogit_icscore = np.zeros(block_size)
    ave_iclogit_icscore = np.zeros(block_size)

    for trial in range(num_tries):
        idx = get_batch().to(device)
        with torch.no_grad():
            logits, _ = model(idx, block_size=block_size, return_full_logits=True)
        is_correct = (torch.argmax(logits[0, block_size:2*block_size, :], dim=1) == idx[0, block_size+1:2*block_size+1])

        for j in range(block_size, 2 * block_size):
            loc = j - block_size
            next_num = idx[0, j + 1].item()
            attn = model.transformer.h[1].attn.stored_attn
            max_score = float('-inf')
            max_score_num = -1
            for k in range(0, 2 * block_size + 1):
                score = attn[j, k].item()
                if score > max_score:
                    max_score = score
                    max_score_num = idx[0, k].item()
            score_correct = (max_score_num == next_num)
            logit_correct = is_correct[loc].item()

            if logit_correct and not score_correct:
                ave_clogit_icscore[loc] += 1.0
            elif not logit_correct and not score_correct:
                ave_iclogit_icscore[loc] += 1.0

    ave_clogit_icscore /= num_tries
    ave_iclogit_icscore /= num_tries
    results[wlnorm] = (ave_clogit_icscore, ave_iclogit_icscore)
    total_ic = np.mean(ave_clogit_icscore + ave_iclogit_icscore)
    print(f'{wlnorm} LN: fraction incorrect scores = {total_ic:.4f}')

clogit_ic_without, iclogit_ic_without = results['without']
clogit_ic_with, iclogit_ic_with = results['with']

total_icscore_without = np.mean(clogit_ic_without + iclogit_ic_without)
total_icscore_with = np.mean(clogit_ic_with + iclogit_ic_with)
epsilon = 1e-10
correction_without = np.sum(clogit_ic_without) / (np.sum(clogit_ic_without + iclogit_ic_without) + epsilon)
correction_with = np.sum(clogit_ic_with) / (np.sum(clogit_ic_with + iclogit_ic_with) + epsilon)

os.makedirs('plots_comparison', exist_ok=True)
fig, ax = plt.subplots(figsize=(5, 4.2))
bar_width = 0.32
x = np.array([0, 1])

bars1 = ax.bar(x[0] - bar_width / 2, total_icscore_without, bar_width, color='#6a3d9a', label='Without LayerNorm')
bars2 = ax.bar(x[0] + bar_width / 2, total_icscore_with, bar_width, color='#e6850e', label='With LayerNorm')
bars3 = ax.bar(x[1] - bar_width / 2, correction_without, bar_width, color='#6a3d9a')
bars4 = ax.bar(x[1] + bar_width / 2, correction_with, bar_width, color='#e6850e')

for bar in [bars1, bars2, bars3, bars4]:
    for b in bar:
        height = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, height + 0.008,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(['Fraction of\nincorrect scores', 'Logit correction ratio\namong incorrect scores'], fontsize=11)
ax.set_ylabel('Fraction', fontsize=12)
ax.set_title('Incorrect scores & logit correction — Layer 1 (Nathan)', fontsize=13, fontweight='bold')
ax.legend(fontsize=10, framealpha=0.9)
ax.grid(True, axis='y', alpha=0.2, linestyle=':')
ax.tick_params(labelsize=11)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylim(0, max(total_icscore_without, total_icscore_with, correction_without, correction_with) * 1.18)
fig.tight_layout()

output_path = 'plots_comparison/compare_cinclogits_nathan_layer1.png'
fig.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()
print(f'Plot saved to {output_path}')
print(f'\nFraction of ICScores — Without LN: {total_icscore_without:.4f}, With LN: {total_icscore_with:.4f}')
print(f'Correction ratio     — Without LN: {correction_without:.4f}, With LN: {correction_with:.4f}')
