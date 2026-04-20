#!/usr/bin/env python3
"""
Plot per-token accuracy (without attn2) as a function of current sorted value i,
for three representative gaps (g=1, g=10, g=50) as separate panels in one row.

Each panel shows individual per-i dots, a rolling-average curve, and a 95%
bootstrap confidence band.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'sortgpt_toolkit'))

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

from model import DEVICE, load_model_from_checkpoint

CKPT = os.path.join(os.path.dirname(__file__), '..', '..', 'new-grid',
                     'k32_N512', 'checkpoints', 'std0p01_iseed1__ckpt100000.pt')
OUTDIR = os.path.join(os.path.dirname(__file__), 'plots')
GAPS = [1, 10, 50]
N_BATCHES = 15000
SMOOTH_WINDOW = 20
N_BOOTSTRAP = 1000
CI_LEVEL = 0.95


def get_batch(vocab_n, block_size, device):
    x = torch.randperm(vocab_n)[:block_size]
    vals, _ = torch.sort(x)
    sep = torch.tensor([vocab_n])
    return torch.cat((x, sep, vals)).unsqueeze(0).to(device)


@torch.no_grad()
def collect_data(model, n_batches):
    bs = model.config.block_size
    vn = model.config.vocab_size - 1
    b0, b1 = model.transformer.h[0], model.transformer.h[1]

    records = []
    for _ in range(n_batches):
        idx = get_batch(vn, bs, DEVICE)
        B, T = idx.size()
        sorted_vals = idx[0, bs + 1:]
        targets = sorted_vals[1:]

        pos = model.transformer.wpe(model.pos_idx[:T])
        embed = model.transformer.wte(idx) + pos
        x = b0(embed)
        x_no_a2 = x + b1.mlp(b1.ln_2(x)) if b1.mlp is not None else x
        x_no_a2 = model.transformer.ln_f(x_no_a2)
        logits = x_no_a2 @ model.lm_head.weight.T
        preds = logits[0, bs + 1:2*bs].argmax(dim=-1)

        correct = (preds == targets).cpu().numpy()
        current_vals = sorted_vals[:-1].cpu().numpy()
        next_vals = targets.cpu().numpy()
        gaps = next_vals - current_vals

        for cv, g, c in zip(current_vals, gaps, correct):
            records.append((int(cv), int(g), int(c)))

    return records


def compute_per_i(records, gap, vocab_n):
    """Raw per-i accuracy (for dots)."""
    by_i = defaultdict(list)
    for cv, g, c in records:
        if g == gap:
            by_i[cv].append(c)
    xs, accs, counts = [], [], []
    for i in sorted(by_i.keys()):
        if len(by_i[i]) >= 3:
            xs.append(i)
            accs.append(np.mean(by_i[i]))
            counts.append(len(by_i[i]))
    return np.array(xs), np.array(accs), np.array(counts)


def smooth_with_ci(records, gap, vocab_n, window=SMOOTH_WINDOW,
                   n_boot=N_BOOTSTRAP, ci=CI_LEVEL, min_pool=10):
    by_i = defaultdict(list)
    for cv, g, c in records:
        if g == gap:
            by_i[cv].append(c)

    smoothed = np.full(vocab_n, np.nan)
    lo = np.full(vocab_n, np.nan)
    hi = np.full(vocab_n, np.nan)

    for center in range(vocab_n):
        start = max(0, center - window // 2)
        end = min(vocab_n, center + window // 2 + 1)
        pool = []
        for j in range(start, end):
            if j in by_i:
                pool.extend(by_i[j])
        if len(pool) < min_pool:
            continue
        arr = np.array(pool, dtype=float)
        smoothed[center] = arr.mean()
        boots = np.array([np.random.choice(arr, size=len(arr), replace=True).mean()
                          for _ in range(n_boot)])
        alpha = (1 - ci) / 2
        lo[center] = np.percentile(boots, 100 * alpha)
        hi[center] = np.percentile(boots, 100 * (1 - alpha))

    xs = np.arange(vocab_n)
    mask = ~np.isnan(smoothed)
    return xs[mask], smoothed[mask], lo[mask], hi[mask]


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    print(f"Loading model from {CKPT}")
    model = load_model_from_checkpoint(CKPT)
    bs = model.config.block_size
    vn = model.config.vocab_size - 1
    print(f"  block_size={bs}, vocab_n={vn}")

    print(f"Collecting data over {N_BATCHES} batches...")
    records = collect_data(model, N_BATCHES)
    print(f"  Collected {len(records)} position records")

    accent_colors = ['#2166ac', '#d6604d', '#4daf4a']
    dot_colors    = ['#92c5de', '#f4a582', '#a6d96a']

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)

    for ax, gap, ac, dc in zip(axes, GAPS, accent_colors, dot_colors):
        xs_dot, accs_dot, counts = compute_per_i(records, gap, vn)
        xs_sm, sm, lo_ci, hi_ci = smooth_with_ci(records, gap, vn)

        n_pts = sum(1 for _, g, _ in records if g == gap)
        print(f"  gap={gap}: {n_pts} data points")

        ax.scatter(xs_dot, accs_dot, s=6, alpha=0.35, color=dc,
                   edgecolors='none', rasterized=True, zorder=1)

        ax.fill_between(xs_sm, lo_ci, hi_ci, color=ac, alpha=0.18, zorder=2)
        ax.plot(xs_sm, sm, color=ac, linewidth=2.2, zorder=3,
                label='Rolling avg')

        ax.set_title(f'$g = {gap}$', fontsize=15, fontweight='bold', pad=8)
        ax.set_xlabel('Current sorted value $i$', fontsize=12)
        ax.set_xlim(0, vn)
        ax.set_ylim(-0.03, 1.06)
        ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
        ax.tick_params(labelsize=10)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes[0].set_ylabel('Per-token accuracy without attn2', fontsize=12)

    fig.suptitle('Accuracy without attn2, by current value $i$ and gap $g$',
                 fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()

    out_path = os.path.join(OUTDIR, 'no_attn2_acc_by_gap.png')
    fig.savefig(out_path, dpi=250, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == '__main__':
    main()
