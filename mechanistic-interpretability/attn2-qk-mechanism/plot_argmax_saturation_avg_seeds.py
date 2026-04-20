#!/usr/bin/env python3
"""
Argmax saturation plot (z=250) averaged across all 5 seeds of k32_N512,
with mean curve and shaded confidence interval (mean ± 1 std).
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'sortgpt_toolkit'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

from model import DEVICE, load_model_from_checkpoint

OUTDIR = os.path.join(os.path.dirname(__file__), 'plots')
os.makedirs(OUTDIR, exist_ok=True)

BASE = os.path.join(os.path.dirname(__file__), '..', '..')

def _ckpt(seed):
    if seed == 1:
        return os.path.join(BASE, 'new-grid', 'k32_N512', 'checkpoints',
                            'std0p01_iseed1__ckpt100000.pt')
    elif seed <= 5:
        return os.path.join(BASE, 'new-grid-multiple', 'k32_N512',
                            f'seed{seed}', 'checkpoints',
                            f'std0p01_iseed{seed}__ckpt100000.pt')
    elif seed <= 15:
        return os.path.join(BASE, 'new-grid-multiple-2', 'k32_N512',
                            f'seed{seed}', 'checkpoints',
                            f'std0p01_iseed{seed}__ckpt100000.pt')
    else:
        return os.path.join(BASE, 'new-grid-multiple-3', 'k32_N512',
                            f'seed{seed}', 'checkpoints',
                            f'std0p01_iseed{seed}__ckpt100000.pt')

SEED_CKPTS = [(f'seed {s}', _ckpt(s)) for s in range(1, 26)]

Z_REF = 250
Y_REF = 250

plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
    'font.size': 13,
    'axes.labelsize': 14,
    'axes.titlesize': 15,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.15,
    'grid.linewidth': 0.5,
    'axes.linewidth': 0.8,
})


@torch.no_grad()
def compute_argmax_curve(ckpt_path, z_ref=Z_REF, y_ref=Y_REF):
    """Load a checkpoint, compute argmax_t score(x, t) for x in [z_ref, z_ref+260)."""
    model = load_model_from_checkpoint(ckpt_path)
    n_embd = model.config.n_embd
    vocab_n = model.config.vocab_size - 1
    block0, block1 = model.transformer.h[0], model.transformer.h[1]

    e_all = model.transformer.wte.weight[:vocab_n]
    ln1_e = block0.ln_1(e_all)
    W_v = block0.attn.c_attn.weight[2*n_embd:3*n_embd, :]
    b_v = block0.attn.c_attn.bias[2*n_embd:3*n_embd]
    v = ln1_e @ W_v.T + b_v
    V_all = v @ block0.attn.c_proj.weight.T + block0.attn.c_proj.bias

    W_q_L2 = block1.attn.c_attn.weight[:n_embd, :]
    b_q_L2 = block1.attn.c_attn.bias[:n_embd]
    W_k_L2 = block1.attn.c_attn.weight[n_embd:2*n_embd, :]
    b_k_L2 = block1.attn.c_attn.bias[n_embd:2*n_embd]

    def compute_Q_all(z):
        inp = e_all[z].unsqueeze(0) + V_all
        mlp_out = block0.mlp(block0.ln_2(inp))
        h = block1.ln_1(mlp_out)
        return h @ W_q_L2.T + b_q_L2

    def compute_K_all(y):
        inp = e_all + V_all[y].unsqueeze(0)
        mlp_out = block0.mlp(block0.ln_2(inp))
        h = block1.ln_1(mlp_out)
        return h @ W_k_L2.T + b_k_L2

    Q = compute_Q_all(z_ref)
    K = compute_K_all(y_ref)
    hm = (Q @ K.T).cpu().numpy()

    x_range = list(range(z_ref, min(vocab_n, z_ref + 260), 2))
    argmax_ts = [int(np.argmax(hm[x, :])) for x in x_range]

    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return np.array(x_range), np.array(argmax_ts)


def main():
    all_argmax = []
    seed_names_used = []
    x_range_ref = None

    for seed_name, ckpt in SEED_CKPTS:
        print(f"Processing {seed_name} ...", flush=True)
        x_range, argmax_ts = compute_argmax_curve(ckpt)
        if x_range_ref is None:
            x_range_ref = x_range
        print(f"  Done. argmax range: [{argmax_ts.min()}, {argmax_ts.max()}]")
        # Filter out seeds with anomalous QK mechanism (argmax drops far below z)
        if argmax_ts.min() < Z_REF - 50:
            print(f"  *** Excluding {seed_name} (outlier: min argmax={argmax_ts.min()})")
            continue
        all_argmax.append(argmax_ts)
        seed_names_used.append(seed_name)

    all_argmax = np.array(all_argmax)
    mean_argmax = np.mean(all_argmax, axis=0)
    std_argmax = np.std(all_argmax, axis=0)
    n_seeds = len(all_argmax)
    print(f"\nUsing {n_seeds} seeds (excluded {len(SEED_CKPTS) - n_seeds} outliers)")

    fig, ax = plt.subplots(figsize=(6, 4.2))

    MAIN_COLOR = '#c0392b'
    BAND_COLOR = '#e74c3c'

    ax.fill_between(x_range_ref, mean_argmax - std_argmax, mean_argmax + std_argmax,
                    color=BAND_COLOR, alpha=0.22, zorder=2, label='$\\pm 1\\,\\sigma$')
    ax.plot(x_range_ref, mean_argmax, '-', color=MAIN_COLOR, linewidth=2.2,
            label=f'Mean ({n_seeds} seeds)', zorder=3)

    ax.plot(x_range_ref, x_range_ref, '--', color='#7f8c8d', linewidth=1.0,
            alpha=0.6, label='$t = x$', zorder=1)
    ax.axhline(Z_REF, color='#95a5a6', linewidth=0.8, linestyle=':', alpha=0.6,
               label=f'$z = {Z_REF}$', zorder=1)

    y_lo = Z_REF - 10
    y_hi = max(mean_argmax + std_argmax) + 15
    ax.set_ylim(y_lo, y_hi)

    ax.set_xlabel('Query-side attn1 target ($x$)')
    ax.set_ylabel('$\\mathrm{argmax}_t\\; s(z,\\, x,\\, t,\\, y)$')
    ax.legend(loc='lower right', frameon=True, framealpha=0.85,
              edgecolor='#cccccc', fancybox=False)

    fig.tight_layout(pad=0.8)
    out_path = os.path.join(OUTDIR, 'argmax_saturation_z250_avg_seeds.png')
    fig.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\nSaved {out_path}")


if __name__ == '__main__':
    main()
