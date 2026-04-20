#!/usr/bin/env python3
"""
Attn spread vs sequence length for k32_N256: sweep K from 2 to 256,
average over all output positions and many inputs.
Three curves: leap-former attn1, leap-former attn2, single-stage attn1.
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

THRESH = 0.04
N_TRIALS = 200
K_VALUES = list(range(2, 257))


def collect_spread_vs_K(ckpt_path, layers_to_collect):
    """For each K, average vdist and nkeys over all positions and trials."""
    model = load_model_from_checkpoint(ckpt_path, extended_max_seq_len=513)
    enable_attention_storage(model)
    vocab_n = model.config.vocab_size - 1

    results = {l: {'vdist': [], 'nkeys': []} for l in layers_to_collect}

    for ki, K in enumerate(K_VALUES):
        per_layer_vd = {l: [] for l in layers_to_collect}
        per_layer_nk = {l: [] for l in layers_to_collect}

        for trial in range(N_TRIALS):
            with torch.no_grad():
                idx = get_batch(1, K, DEVICE, vocab_n=vocab_n)
                model(idx, block_size=K)
                tokens = idx[0].cpu().numpy()
                unsorted = tokens[:K]
                sorted_t = tokens[K + 1:]

                for layer in layers_to_collect:
                    attn = model.transformer.h[layer].attn.attn.cpu().numpy()
                    for p in range(K):
                        qp = K + p
                        qval = sorted_t[p] if p < len(sorted_t) else tokens[qp]
                        ua = attn[qp, :K]
                        mask = ua > THRESH
                        if mask.any():
                            attended = unsorted[mask]
                            dists = np.abs(attended.astype(float) - float(qval))
                            per_layer_vd[layer].append(np.mean(dists))
                            per_layer_nk[layer].append(int(mask.sum()))

        for layer in layers_to_collect:
            results[layer]['vdist'].append(
                np.mean(per_layer_vd[layer]) if per_layer_vd[layer] else np.nan)
            results[layer]['nkeys'].append(
                np.mean(per_layer_nk[layer]) if per_layer_nk[layer] else np.nan)

        if (ki + 1) % 50 == 0:
            print(f"  K={K} done ({ki+1}/{len(K_VALUES)})")

    return results


print("Collecting from leap-former (seed 1, k32_N256) ...")
leap_data = collect_spread_vs_K(CKPT_LEAP, [0, 1])

print("Collecting from single-stage (seed 4, k32_N256) ...")
single_data = collect_spread_vs_K(CKPT_SINGLE, [0])

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

styles = [
    ('leap', 0, '#4C72B0', '-',  'Attn1 — two-stage model'),
    ('leap', 1, '#C44E52', '-',  'Attn2 — two-stage model'),
    ('single', 0, '#55A868', '--', 'Attn1 — single-stage model'),
]

# Left: average numerical distance
ax = axes[0]
for tag, layer, color, ls, label in styles:
    data = leap_data if tag == 'leap' else single_data
    ax.plot(K_VALUES, data[layer]['vdist'], color=color, linestyle=ls,
            linewidth=2, label=label, alpha=0.85)
ax.axvline(32, color='#555555', linestyle='--', linewidth=1.5, alpha=0.7, zorder=0)
ax.set_xlabel('Sequence length $K$')
ax.set_ylabel('Avg |query value $-$ key value|')
ax.set_title(f'Average Numerical Distance to Attended Keys\n(attn threshold $> {THRESH}$, averaged over positions)')
ax.legend(fontsize=9)

# Right: candidate set size
ax = axes[1]
for tag, layer, color, ls, label in styles:
    data = leap_data if tag == 'leap' else single_data
    ax.plot(K_VALUES, data[layer]['nkeys'], color=color, linestyle=ls,
            linewidth=2, label=label, alpha=0.85)
ax.axvline(32, color='#555555', linestyle='--', linewidth=1.5, alpha=0.7, zorder=0)
ax.set_xlabel('Sequence length $K$')
ax.set_ylabel('Avg number of attended keys')
ax.set_title(f'Candidate Set Size\n(attn threshold $> {THRESH}$, averaged over positions)')
ax.legend(fontsize=9)

_ann_kw = dict(fontsize=8, color='#555555', ha='left', va='top',
               arrowprops=dict(arrowstyle='->', color='#555555', lw=1.0),
               bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='#bbbbbb', alpha=0.85))
for a in axes:
    yl = a.get_ylim()
    a.annotate('Training\nlength', xy=(32, yl[1]*0.92), xytext=(50, yl[1]*0.80), **_ann_kw)

fig.tight_layout()
outpath = os.path.join(OUTDIR, 'attn_spread_vs_seqlen_k32_N256.png')
fig.savefig(outpath, dpi=200, bbox_inches='tight')
plt.close()
print(f"Saved {outpath}")

# Separate plots
for metric, ylabel, title_base, suffix in [
    ('vdist', 'Avg |query value $-$ key value|',
     'Average Numerical Distance to Attended Keys', 'vdist'),
    ('nkeys', 'Avg number of attended keys',
     'Candidate Set Size', 'nkeys'),
]:
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    for tag, layer, color, ls, label in styles:
        data = leap_data if tag == 'leap' else single_data
        ax2.plot(K_VALUES, data[layer][metric], color=color, linestyle=ls,
                 linewidth=2, label=label, alpha=0.85)
    ax2.axvline(32, color='#555555', linestyle='--', linewidth=1.5, alpha=0.7, zorder=0)
    ax2.set_xlabel('Sequence length $K$')
    ax2.set_ylabel(ylabel)
    ax2.set_title(f'{title_base}\n(attn threshold $> {THRESH}$, averaged over positions)')
    yl2 = ax2.get_ylim()
    ax2.annotate('Training\nlength', xy=(32, yl2[1]*0.92), xytext=(50, yl2[1]*0.80),
                 fontsize=9, color='#555555', ha='left', va='top',
                 arrowprops=dict(arrowstyle='->', color='#555555', lw=1.0),
                 bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#bbbbbb', alpha=0.85))
    ax2.legend(fontsize=10)
    fig2.tight_layout()
    sep_path = os.path.join(OUTDIR, f'attn_{suffix}_vs_seqlen_k32_N256.png')
    fig2.savefig(sep_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved {sep_path}")
