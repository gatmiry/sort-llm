#!/usr/bin/env python3
"""
Attn spread comparison for k32_N256: leap-former (seed 1) attn1+attn2
overlaid with single-stage (seed 4) attn1.
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

N_TRIALS = 1000
THRESH = 0.04


def collect_spread(ckpt_path, layers_to_collect):
    """Collect vdist and nkeys per output position for specified layers."""
    model = load_model_from_checkpoint(ckpt_path)
    enable_attention_storage(model)
    block_size = model.config.block_size
    vocab_n = model.config.vocab_size - 1

    vdist = {l: [[] for _ in range(block_size)] for l in layers_to_collect}
    nkeys = {l: [[] for _ in range(block_size)] for l in layers_to_collect}

    for trial in range(N_TRIALS):
        with torch.no_grad():
            idx = get_batch(1, block_size, DEVICE, vocab_n=vocab_n)
            model(idx, block_size=block_size)
            tokens = idx[0].cpu().numpy()
            unsorted = tokens[:block_size]
            sorted_t = tokens[block_size + 1:]

            for layer in layers_to_collect:
                attn = model.transformer.h[layer].attn.attn.cpu().numpy()
                for p in range(block_size):
                    qp = block_size + p
                    qval = sorted_t[p] if p < len(sorted_t) else tokens[qp]
                    ua = attn[qp, :block_size]
                    mask = ua > THRESH
                    if mask.any():
                        attended = unsorted[mask]
                        dists = np.abs(attended.astype(float) - float(qval))
                        vdist[layer][p].append(np.mean(dists))
                        nkeys[layer][p].append(int(mask.sum()))

        if (trial + 1) % 250 == 0:
            print(f"  {trial + 1}/{N_TRIALS}")

    result = {}
    for layer in layers_to_collect:
        means_vd, means_nk, positions = [], [], []
        for p in range(block_size):
            if vdist[layer][p]:
                means_vd.append(np.mean(vdist[layer][p]))
                means_nk.append(np.mean(nkeys[layer][p]))
                positions.append(p)
        result[layer] = (positions, means_vd, means_nk)
    return result


print("Collecting from leap-former (seed 1, k32_N256) ...")
leap_data = collect_spread(CKPT_LEAP, [0, 1])

print("Collecting from single-stage (seed 4, k32_N256) ...")
single_data = collect_spread(CKPT_SINGLE, [0])

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

styles = [
    ('leap', 0, '#4C72B0', '-',  'Attn1 — leap-former (seed 1)'),
    ('leap', 1, '#C44E52', '-',  'Attn2 — leap-former (seed 1)'),
    ('single', 0, '#55A868', '--', 'Attn1 — single-stage (seed 4)'),
]

# Left: average numerical distance
ax = axes[0]
for tag, layer, color, ls, label in styles:
    data = leap_data if tag == 'leap' else single_data
    pos, vd, _ = data[layer]
    ax.plot(pos, vd, color=color, linestyle=ls, linewidth=2, label=label)
ax.set_xlabel('Sorted output position index')
ax.set_ylabel('Avg |query value $-$ key value|')
ax.set_title(f'Average Numerical Distance to Attended Keys\n(attn threshold $> {THRESH}$)')
ax.legend()

# Right: candidate set size
ax = axes[1]
for tag, layer, color, ls, label in styles:
    data = leap_data if tag == 'leap' else single_data
    pos, _, nk = data[layer]
    ax.plot(pos, nk, color=color, linestyle=ls, linewidth=2, label=label)
ax.set_xlabel('Sorted output position index')
ax.set_ylabel('Avg number of attended keys')
ax.set_title(f'Candidate Set Size\n(attn threshold $> {THRESH}$)')
ax.legend()

fig.tight_layout()
outpath = os.path.join(OUTDIR, 'attn_spread_k32_N256_leap_vs_single.png')
fig.savefig(outpath, dpi=200, bbox_inches='tight')
plt.close()
print(f"Saved {outpath}")
