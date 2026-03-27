"""
Generate the Q-K interaction scores plot for the paper.

Run from the sort-llm root directory:
    python results/qk_interaction_scores/generate_plot.py
"""
import torch
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from model_tbyt_3 import GPT, GPTConfig

checkpoint_path = os.path.join(
    os.path.dirname(__file__), '../../saved_models/2026-03-14_18-40-26_vocab128/'
    'march14-withoutlayernorm-block_size:32-batch_size:4096-n_embd:64_head:1_layers:2_vocab_size:128_itr:80000_checkpoint.pt'
)
device = 'cpu'

config = GPTConfig(block_size=32, vocab_size=128, without_pos=True)
config.n_embd = 64
config.n_heads = 1
config.n_layers = 2

model = GPT(config)
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model'])
model.eval()

qweights, kweights, vweights = model.transformer.h[0].c_attn.c_attn.weight.split(model.config.n_embd, dim=0)

scores = (model.transformer.wte.weight @ qweights.t() @ kweights @ model.transformer.wte.weight.t()).detach().numpy()

rows = [10, 30, 50, 70, 90, 110]
fig, axes = plt.subplots(1, 6, figsize=(18, 3.2), sharey=True)

for ax, row in zip(axes, rows):
    ax.plot(scores[row, :], linewidth=2.2, color='#2c7bb6')
    ax.axvline(x=row, color='#d7191c', linestyle='--', linewidth=2.0, alpha=0.85)
    ax.fill_between(range(scores.shape[1]), scores[row, :], alpha=0.08, color='#2c7bb6')
    ax.set_title(f'embedding of {row}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Token index (j)', fontsize=13)
    ax.grid(True, alpha=0.2, linestyle=':')
    ax.tick_params(labelsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

for ax, row in zip(axes, rows):
    ax.set_xticks([row])
    ax.set_xticklabels([str(row)], fontweight='bold', color='#d7191c', fontsize=12)

axes[0].set_ylabel('Score', fontsize=14)
plt.tight_layout()

output_path = os.path.join(os.path.dirname(__file__), 'qk_interaction_scores.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f'Plot saved to {output_path}')
