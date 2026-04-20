#!/usr/bin/env python3
"""
Attention top-1 accuracy on the correct next token vs sequence length K
for k32_N256: two-stage (seed 1) attn1+attn2, single-stage (seed 4) attn1.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'sortgpt_toolkit'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from model import DEVICE, load_model_from_checkpoint, get_batch
from intervene import enable_attention_storage

BASE = os.path.join(os.path.dirname(__file__), '..', '..')
CKPT_LEAP = os.path.join(BASE, 'new-grid', 'k32_N256',
                          'checkpoints', 'std0p01_iseed1__ckpt100000.pt')
CKPT_SINGLE = os.path.join(BASE, 'new-grid-multiple', 'k32_N256', 'seed4',
                            'checkpoints', 'std0p01_iseed4__ckpt100000.pt')
OUTDIR = os.path.join(os.path.dirname(__file__), 'plots')
os.makedirs(OUTDIR, exist_ok=True)

PAPER_RC = {
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.2,
}
plt.rcParams.update(PAPER_RC)

N_TRIALS = 200
K_VALUES = list(range(2, 257))


def collect_top1_vs_K(ckpt_path, layers_to_collect):
    """For each K, compute fraction of times argmax(attn) = correct next token position."""
    model = load_model_from_checkpoint(ckpt_path, extended_max_seq_len=513)
    enable_attention_storage(model)
    vocab_n = model.config.vocab_size - 1

    results = {l: {'acc': []} for l in layers_to_collect}
    avg_gaps = []

    for ki, K in enumerate(K_VALUES):
        per_layer_correct = {l: 0 for l in layers_to_collect}
        per_layer_total = {l: 0 for l in layers_to_collect}
        trial_gaps = []

        for trial in range(N_TRIALS):
            with torch.no_grad():
                idx = get_batch(1, K, DEVICE, vocab_n=vocab_n)
                model(idx, block_size=K)
                tokens = idx[0].cpu().numpy()
                unsorted = tokens[:K]
                sorted_t = tokens[K + 1:]

                if K > 1:
                    gaps = np.diff(sorted_t.astype(float))
                    trial_gaps.append(np.mean(gaps))

                val_to_pos = {}
                for pos in range(K):
                    val_to_pos[int(unsorted[pos])] = pos

                for layer in layers_to_collect:
                    attn = model.transformer.h[layer].attn.attn.cpu().numpy()
                    for p in range(K - 1):
                        qp = K + 1 + p
                        target = int(sorted_t[p + 1])
                        if target not in val_to_pos:
                            continue
                        correct_pos = val_to_pos[target]
                        attended_pos = np.argmax(attn[qp, :K])
                        if attended_pos == correct_pos:
                            per_layer_correct[layer] += 1
                        per_layer_total[layer] += 1

        for layer in layers_to_collect:
            acc = per_layer_correct[layer] / per_layer_total[layer] if per_layer_total[layer] > 0 else np.nan
            results[layer]['acc'].append(acc)
        avg_gaps.append(np.mean(trial_gaps) if trial_gaps else np.nan)

        if (ki + 1) % 50 == 0:
            print(f"  K={K} done ({ki+1}/{len(K_VALUES)})")

    return results, avg_gaps


print("Collecting from two-stage model (seed 1, k32_N256) ...")
leap_data, avg_gaps = collect_top1_vs_K(CKPT_LEAP, [0, 1])

print("Collecting from single-stage model (seed 4, k32_N256) ...")
single_data, _ = collect_top1_vs_K(CKPT_SINGLE, [0])

fig, ax = plt.subplots(figsize=(8, 5))

styles = [
    ('leap', 0, '#4C72B0', '-',  'Attn1 — two-stage model'),
    ('leap', 1, '#C44E52', '-',  'Attn2 — two-stage model'),
    ('single', 0, '#55A868', '--', 'Attn1 — single-stage model'),
]

for tag, layer, color, ls, label in styles:
    data = leap_data if tag == 'leap' else single_data
    ax.plot(K_VALUES, data[layer]['acc'], color=color, linestyle=ls,
            linewidth=2, label=label, alpha=0.85)

ax.axvline(32, color='#555555', linestyle='--', linewidth=1.5, alpha=0.7, zorder=0)
ax.annotate('Training\nlength', xy=(32, 0.98), xytext=(44, 0.88),
            fontsize=9, color='#555555', ha='left', va='top',
            arrowprops=dict(arrowstyle='->', color='#555555', lw=1.2),
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#bbbbbb', alpha=0.85))
ax.set_xlabel('Sequence length $K$')
ax.set_ylabel('Top-1 accuracy (fraction correct)')
ax.set_title('Attention Top-1 Accuracy on Correct Next Token\n(averaged over positions)')
ax.set_ylim(-0.02, 1.02)

ax2 = ax.twinx()
ax2.plot(K_VALUES, avg_gaps, color='#999999', linestyle='-', linewidth=1.5,
         alpha=0.7, label='Avg consecutive gap')
ax2.set_ylabel('Avg gap between consecutive sorted values', color='#999999')
ax2.tick_params(axis='y', labelcolor='#999999')
ax2.spines['right'].set_visible(True)
ax2.spines['right'].set_color('#999999')

lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc='center right')

fig.tight_layout()
outpath = os.path.join(OUTDIR, 'attn_top1_vs_seqlen_k32_N256.png')
fig.savefig(outpath, dpi=200, bbox_inches='tight')
plt.close()
print(f"Saved {outpath}")
