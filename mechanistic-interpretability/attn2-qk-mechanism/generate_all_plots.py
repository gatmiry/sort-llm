#!/usr/bin/env python3
"""
Generate all paper-quality figures for the attn2 QK mechanism subsection.
Checkpoint: k32_N512, seed 1 (new-grid), ckpt100000.

Figures produced:
  1. attn_error_rates.png        - Attn1 vs Attn2 top-1 error rate comparison
  2. qk_heatmap_asymmetry.png    - QK heatmaps: query-side vs key-side L1 context
  3. qk_score_slices_band.png    - QK score slices showing windowed monotonicity
  4. argmax_saturation.png        - Attn2 argmax caps under extreme L1 targets
  5. l1_vs_l2_qk_smoothness.png  - L1 vs L2 QK score overlay showing L2 smoothness
  6. attn_spread_comparison.png   - Candidate set size and distance for L1 vs L2
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'sortgpt_toolkit'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from model import DEVICE, load_model_from_checkpoint, get_batch
from intervene import enable_attention_storage

CKPT = os.path.join(os.path.dirname(__file__), '..', '..', 'new-grid', 'k32_N512',
                     'checkpoints', 'std0p01_iseed1__ckpt100000.pt')
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

model = load_model_from_checkpoint(CKPT)
n_embd = model.config.n_embd
vocab_n = model.config.vocab_size - 1
block_size = model.config.block_size
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

W_q_L1 = block0.attn.c_attn.weight[:n_embd, :]
b_q_L1 = block0.attn.c_attn.bias[:n_embd]
W_k_L1 = block0.attn.c_attn.weight[n_embd:2*n_embd, :]
b_k_L1 = block0.attn.c_attn.bias[n_embd:2*n_embd]


@torch.no_grad()
def compute_Q_all(z):
    """Q vectors for attn2: all possible L1 targets x, fixed query token z."""
    inp = e_all[z].unsqueeze(0) + V_all
    mlp_out = block0.mlp(block0.ln_2(inp))
    h = block1.ln_1(mlp_out)
    return h @ W_q_L2.T + b_q_L2

@torch.no_grad()
def compute_K_all(y):
    """K vectors for attn2: all key tokens t, fixed key-side L1 target y."""
    inp = e_all + V_all[y].unsqueeze(0)
    mlp_out = block0.mlp(block0.ln_2(inp))
    h = block1.ln_1(mlp_out)
    return h @ W_k_L2.T + b_k_L2

@torch.no_grad()
def compute_heatmap_xt(z, y):
    """Attn2 QK score heatmap: score[x, t] for fixed (z, y)."""
    Q = compute_Q_all(z)
    K = compute_K_all(y)
    return (Q @ K.T).cpu().numpy()


# ====================================================================
# FIGURE 1: Attn1 vs Attn2 error rates
# ====================================================================
def fig1_attn_error_rates():
    print("Fig 1: Attention error rates ...")
    enable_attention_storage(model)
    N_TRIALS = 3000
    l1_errors, l2_errors = 0, 0
    total = 0

    for trial in range(N_TRIALS):
        with torch.no_grad():
            idx = get_batch(1, block_size, DEVICE, vocab_n=vocab_n)
            model(idx, block_size=block_size)
            tokens = idx[0].cpu().numpy()
            unsorted = tokens[:block_size]
            sorted_t = tokens[block_size+1:]

            val_to_pos = {}
            for p in range(block_size):
                val_to_pos[int(unsorted[p])] = p

            for layer_idx, layer_name in [(0, 'L1'), (1, 'L2')]:
                attn = model.transformer.h[layer_idx].attn.attn.cpu().numpy()
                for p in range(block_size - 1):
                    qp = block_size + 1 + p
                    target = int(sorted_t[p + 1])
                    if target not in val_to_pos:
                        continue
                    correct_key_pos = val_to_pos[target]
                    attended_pos = np.argmax(attn[qp, :block_size])
                    total_here = 1
                    if attended_pos != correct_key_pos:
                        if layer_idx == 0:
                            l1_errors += 1
                        else:
                            l2_errors += 1
                    total += total_here if layer_idx == 0 else 0

    l1_rate = l1_errors / total if total > 0 else 0
    l2_rate = l2_errors / total if total > 0 else 0
    print(f"  Attn1 error: {l1_rate:.3f}, Attn2 error: {l2_rate:.3f}")

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(['Attn1 (Layer 1)', 'Attn2 (Layer 2)'], [l1_rate, l2_rate],
                  color=['#4C72B0', '#C44E52'], width=0.5, edgecolor='white')
    for bar, val in zip(bars, [l1_rate, l2_rate]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.1%}', ha='center', va='bottom', fontweight='bold', fontsize=12)
    ax.set_ylabel('Top-1 Error Rate')
    ax.set_title('Attention Top-1 Error Rate')
    ax.set_ylim(0, max(l1_rate, l2_rate) * 1.25)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, 'attn_error_rates.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved attn_error_rates.png")


# ====================================================================
# FIGURE 2: QK heatmap asymmetry (query-side vs key-side L1 context)
# ====================================================================
def fig2_qk_heatmap_asymmetry():
    print("Fig 2: QK heatmap asymmetry ...")
    z_ref = 250

    # Panel A: Fix (z=250, x=251), vary key-side: y-axis=y (key L1 target), x-axis=t (key token)
    # = image (2) from slack: shows key-side L1 target doesn't change score much
    Q_fixed = compute_Q_all(z_ref)
    q_vec = Q_fixed[z_ref + 1]  # x = z+1 = 251
    # For each y, compute K(t, y) for all t, then score = q · K
    hm_key = np.zeros((vocab_n, vocab_n))
    for y_val in range(vocab_n):
        K = compute_K_all(y_val)
        scores = (q_vec.unsqueeze(0) @ K.T).squeeze(0).cpu().numpy()
        hm_key[y_val, :] = scores

    # Panel B: Fix (z=250, y=250), vary query-side: y-axis=x (query L1 target), x-axis=t (key token)
    # = image (3) from slack: shows query-side L1 target controls the band
    hm_query = compute_heatmap_xt(z_ref, z_ref)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    for ax, hm, ylabel, title_suffix in [
        (axes[0], hm_key,
         'Key-side attn1 target ($y$)',
         'Varying key-side attn1 target $y$'),
        (axes[1], hm_query,
         'Query-side attn1 target ($x$)',
         'Varying query-side attn1 target $x$'),
    ]:
        vmax = np.percentile(np.abs(hm), 99)
        im = ax.imshow(hm, aspect='auto', origin='lower', cmap='RdBu_r',
                       vmin=-vmax, vmax=vmax, interpolation='nearest',
                       extent=[0, vocab_n, 0, vocab_n])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='QK score')
        ax.set_xlabel('Key token value ($t$)')
        ax.set_ylabel(ylabel)
        ax.set_title(title_suffix)

    fig.suptitle(f'Attn2 QK Score Heatmaps (query token $z = {z_ref}$)', fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(OUTDIR, 'qk_heatmap_asymmetry.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved qk_heatmap_asymmetry.png")


# ====================================================================
# FIGURE 3: QK score slices showing band + monotonicity
# ====================================================================
def fig3_qk_score_slices_band():
    print("Fig 3: QK score slices (band mechanism) ...")
    z_ref, y_ref = 250, 250
    hm = compute_heatmap_xt(z_ref, y_ref)

    x_slices = [260, 270, 280, 290, 300]
    colors = ['#1b7837', '#2166ac', '#d6604d', '#762a83', '#e08214']
    t_range = np.arange(vocab_n)

    fig, ax = plt.subplots(figsize=(10, 5))
    for x_val, color in zip(x_slices, colors):
        scores = hm[x_val, :]
        amax = np.argmax(scores)
        ax.plot(t_range, scores, color=color, linewidth=1.5, alpha=0.85,
                label=f'$x = {x_val}$ (argmax $t = {amax}$)')
        ax.axvline(amax, color=color, linewidth=0.6, linestyle=':', alpha=0.4)

    ax.axvline(z_ref, color='gray', linewidth=1, linestyle='--', alpha=0.5,
               label=f'$z = {z_ref}$')
    ax.set_xlabel('Key token value ($t$)')
    ax.set_ylabel('Attn2 QK Score (pre-softmax)')
    ax.set_title(f'Attn2 QK Score vs Key Token for Different Attn1 Targets\n'
                 f'(query token $z = {z_ref}$, key-side attn1 target $y = {y_ref}$)')
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, 'qk_score_slices_band.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved qk_score_slices_band.png")


# ====================================================================
# FIGURE 4: Argmax saturation under extreme L1 targets
# ====================================================================
def fig4_argmax_saturation():
    print("Fig 4: Argmax saturation ...")
    z_vals = [100, 250, 400]
    y_ref = 250
    fig, axes = plt.subplots(1, len(z_vals), figsize=(5 * len(z_vals), 4.5))

    for ax, z_ref in zip(axes, z_vals):
        hm = compute_heatmap_xt(z_ref, y_ref)
        x_range = list(range(z_ref, min(vocab_n, z_ref + 260), 2))
        argmax_ts = [int(np.argmax(hm[x, :])) for x in x_range]

        ax.plot(x_range, argmax_ts, 'o-', color='#b2182b', linewidth=1.5, markersize=3)
        ax.plot(x_range, x_range, 'k--', linewidth=0.8, alpha=0.3, label='$t = x$ (diagonal)')
        ax.axhline(z_ref, color='gray', linewidth=0.8, linestyle=':', alpha=0.5,
                    label=f'$z = {z_ref}$')
        ax.set_xlabel('Query-side attn1 target ($x$)')
        ax.set_ylabel('$\\mathrm{argmax}_t\\, \\mathrm{score}(x, t)$')
        ax.set_title(f'$z = {z_ref}$')
        ax.legend(fontsize=8)

    fig.suptitle(f'Attn2 Argmax Saturation Under Increasing Attn1 Targets\n'
                 f'(key-side attn1 target $y = {y_ref}$)', fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(os.path.join(OUTDIR, 'argmax_saturation.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved argmax_saturation.png")


# ====================================================================
# FIGURE 5: L1 vs L2 QK score smoothness comparison
# ====================================================================
def fig5_l1_vs_l2_smoothness():
    print("Fig 5: L1 vs L2 QK smoothness ...")
    ref = 250
    z_ref, y_ref = ref, ref

    h_q_l1 = block0.ln_1(e_all[ref].unsqueeze(0))
    Q_l1 = h_q_l1 @ W_q_L1.T + b_q_L1
    h_k_l1 = block0.ln_1(e_all)
    K_l1 = h_k_l1 @ W_k_L1.T + b_k_L1
    l1_scores = (Q_l1 @ K_l1.T).squeeze(0).detach().cpu().numpy()

    hm = compute_heatmap_xt(z_ref, y_ref)
    x_slices = [260, 280, 300]
    slice_colors = ['#2166ac', '#d6604d', '#762a83']
    t_range = np.arange(vocab_n)

    fig, ax1 = plt.subplots(figsize=(12, 5.5))

    ax1.plot(t_range, l1_scores, color='black', linewidth=2.0, alpha=0.6,
             label=f'Attn1: query $= {ref}$ vs all keys', zorder=10)
    ax1.set_xlabel('Key token value')
    ax1.set_ylabel('Attn1 QK Score', color='black')
    ax1.tick_params(axis='y', labelcolor='black')

    ax2 = ax1.twinx()
    for x_val, color in zip(x_slices, slice_colors):
        scores = hm[x_val, :]
        ax2.plot(t_range, scores, color=color, linewidth=1.3, alpha=0.8,
                 label=f'Attn2: $x = {x_val}$, $z = {z_ref}$')
    ax2.set_ylabel('Attn2 QK Score', color='#2166ac')
    ax2.tick_params(axis='y', labelcolor='#2166ac')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc='upper left')

    ax1.axvline(ref, color='red', linewidth=0.8, linestyle='--', alpha=0.4)
    ax1.set_title(f'Attn1 vs Attn2 QK Scores: Smoothness Comparison\n'
                  f'(query token $z = {z_ref}$, key-side $y = {y_ref}$)')
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, 'l1_vs_l2_qk_smoothness.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved l1_vs_l2_qk_smoothness.png")


# ====================================================================
# FIGURE 6: Attention spread comparison (candidate set size + distance)
# ====================================================================
def fig6_attn_spread():
    print("Fig 6: Attention spread comparison ...")
    enable_attention_storage(model)
    N_TRIALS = 1000
    fixed_thresh = 0.04
    n_layers = 2

    vdist_by_pos = {l: [[] for _ in range(block_size)] for l in range(n_layers)}
    nkeys_by_pos = {l: [[] for _ in range(block_size)] for l in range(n_layers)}

    for trial in range(N_TRIALS):
        with torch.no_grad():
            idx = get_batch(1, block_size, DEVICE, vocab_n=vocab_n)
            model(idx, block_size=block_size)
            tokens = idx[0].cpu().numpy()
            unsorted = tokens[:block_size]
            sorted_t = tokens[block_size+1:]

            for layer in range(n_layers):
                attn = model.transformer.h[layer].attn.attn.cpu().numpy()
                for p in range(block_size):
                    qp = block_size + p
                    qval = sorted_t[p] if p < len(sorted_t) else tokens[qp]
                    ua = attn[qp, :block_size]
                    mask = ua > fixed_thresh
                    if mask.any():
                        attended = unsorted[mask]
                        dists = np.abs(attended.astype(float) - float(qval))
                        vdist_by_pos[layer][p].append(np.mean(dists))
                        nkeys_by_pos[layer][p].append(int(mask.sum()))

        if (trial + 1) % 250 == 0:
            print(f"  {trial+1}/{N_TRIALS}")

    layer_colors = {0: '#4C72B0', 1: '#C44E52'}
    layer_names = {0: 'Attn1 (Layer 1)', 1: 'Attn2 (Layer 2)'}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    for layer in range(n_layers):
        means, positions = [], []
        for p in range(block_size):
            if vdist_by_pos[layer][p]:
                means.append(np.mean(vdist_by_pos[layer][p]))
                positions.append(p)
        ax.plot(positions, means, color=layer_colors[layer], linewidth=2,
                label=layer_names[layer])
    ax.set_xlabel('Sorted output position index')
    ax.set_ylabel('Avg |query value $-$ key value|')
    ax.set_title(f'Average Numerical Distance to Attended Keys\n(attn threshold $> {fixed_thresh}$)')
    ax.legend()

    ax = axes[1]
    for layer in range(n_layers):
        means, positions = [], []
        for p in range(block_size):
            if nkeys_by_pos[layer][p]:
                means.append(np.mean(nkeys_by_pos[layer][p]))
                positions.append(p)
        ax.plot(positions, means, color=layer_colors[layer], linewidth=2,
                label=layer_names[layer])
    ax.set_xlabel('Sorted output position index')
    ax.set_ylabel('Avg number of attended keys')
    ax.set_title(f'Candidate Set Size\n(attn threshold $> {fixed_thresh}$)')
    ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, 'attn_spread_comparison.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved attn_spread_comparison.png")


if __name__ == '__main__':
    fig1_attn_error_rates()
    fig2_qk_heatmap_asymmetry()
    fig3_qk_score_slices_band()
    fig4_argmax_saturation()
    fig5_l1_vs_l2_smoothness()
    fig6_attn_spread()
    print("\nAll figures generated.")
