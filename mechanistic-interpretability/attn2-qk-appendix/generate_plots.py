#!/usr/bin/env python3
"""
Generate split-attention plots for appendix: attn1 equally focused on 2, 3, or 4 tokens.
Also performs argmax bias analysis.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'sortgpt_toolkit'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from model import DEVICE, load_model_from_checkpoint

CKPT = os.path.join(os.path.dirname(__file__), '..', '..', 'new-grid', 'k32_N512',
                     'checkpoints', 'std0p01_iseed1__ckpt100000.pt')
OUTDIR = os.path.join(os.path.dirname(__file__), 'plots')
os.makedirs(OUTDIR, exist_ok=True)

plt.rcParams.update({'figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white',
                     'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':0.2,
                     'font.size':11,'axes.labelsize':12,'axes.titlesize':13})

model = load_model_from_checkpoint(CKPT)
n_embd = model.config.n_embd
vocab_n = model.config.vocab_size - 1
block0, block1 = model.transformer.h[0], model.transformer.h[1]

e_all = model.transformer.wte.weight[:vocab_n]
ln1_e = block0.ln_1(e_all)
W_v = block0.attn.c_attn.weight[2*n_embd:3*n_embd, :]
b_v = block0.attn.c_attn.bias[2*n_embd:3*n_embd]
v = ln1_e @ W_v.T + b_v
V_all = v @ block0.attn.c_proj.weight.T + block0.attn.c_proj.bias

W_q = block1.attn.c_attn.weight[:n_embd, :]
b_q = block1.attn.c_attn.bias[:n_embd]
W_k = block1.attn.c_attn.weight[n_embd:2*n_embd, :]
b_k = block1.attn.c_attn.bias[n_embd:2*n_embd]


@torch.no_grad()
def V_split(token_set):
    """Compute equally-weighted V output for a set of tokens."""
    vs = torch.stack([V_all[t] for t in token_set])
    return vs.mean(dim=0)


@torch.no_grad()
def compute_score_slice(z, query_tokens, key_y_tokens, t_range_max=vocab_n):
    """Score(t) for a single query config: q(z, query_tokens) . k(t, key_y_tokens)."""
    V_q = V_split(query_tokens)
    inp_q = e_all[z].unsqueeze(0) + V_q.unsqueeze(0)
    mlp_q = block0.mlp(block0.ln_2(inp_q))
    q = (block1.ln_1(mlp_q) @ W_q.T + b_q).squeeze(0)

    V_k = V_split(key_y_tokens)
    inp_k = e_all[:t_range_max] + V_k.unsqueeze(0)
    mlp_k = block0.mlp(block0.ln_2(inp_k))
    K = block1.ln_1(mlp_k) @ W_k.T + b_k
    return (q @ K.T).cpu().numpy()


@torch.no_grad()
def compute_heatmap(z, key_y_tokens, n_query_targets=vocab_n):
    """Heatmap[x, t] where x is the *minimum* of a single query-side target (for n_tokens=1)."""
    V_k = V_split(key_y_tokens)
    inp_k = e_all + V_k.unsqueeze(0)
    mlp_k = block0.mlp(block0.ln_2(inp_k))
    K = block1.ln_1(mlp_k) @ W_k.T + b_k

    hm = np.zeros((n_query_targets, vocab_n))
    for x in range(n_query_targets):
        V_q = V_all[x]
        inp_q = e_all[z].unsqueeze(0) + V_q.unsqueeze(0)
        mlp_q = block0.mlp(block0.ln_2(inp_q))
        q = (block1.ln_1(mlp_q) @ W_q.T + b_q).squeeze(0)
        hm[x, :] = (q @ K.T).cpu().numpy()
    return hm


@torch.no_grad()
def compute_heatmap_split(z, key_y_tokens, n_tokens, delta):
    """Heatmap[x, t] where query-side attn1 is equally split on n_tokens tokens: x, x+delta, x+2*delta, ..."""
    V_k = V_split(key_y_tokens)
    inp_k = e_all + V_k.unsqueeze(0)
    mlp_k = block0.mlp(block0.ln_2(inp_k))
    K = block1.ln_1(mlp_k) @ W_k.T + b_k

    max_x = vocab_n - (n_tokens - 1) * delta
    hm = np.zeros((max_x, vocab_n))
    for x in range(max_x):
        tokens = [x + i * delta for i in range(n_tokens)]
        V_q = V_split(tokens)
        inp_q = e_all[z].unsqueeze(0) + V_q.unsqueeze(0)
        mlp_q = block0.mlp(block0.ln_2(inp_q))
        q = (block1.ln_1(mlp_q) @ W_q.T + b_q).squeeze(0)
        hm[x, :] = (q @ K.T).cpu().numpy()
    return hm


# ==========================================================================
# SLICE PLOTS for n_tokens = 2, 3, 4 and various deltas
# ==========================================================================
def generate_slice_plots():
    print("=== Generating slice plots ===")
    z_fixed = 250
    y_tokens = [251]
    x_base_values = [260, 270, 280, 290, 300]
    colors = ['#1b7837', '#2166ac', '#d6604d', '#762a83', '#e08214']
    t_range = np.arange(vocab_n)

    for n_tok in [2, 3, 4]:
        deltas = [1, 5, 20]
        fig, axes = plt.subplots(1, len(deltas), figsize=(6 * len(deltas), 5))
        for ax, delta in zip(axes, deltas):
            for x_base, color in zip(x_base_values, colors):
                q_tokens = [x_base + i * delta for i in range(n_tok)]
                if max(q_tokens) >= vocab_n:
                    continue
                scores = compute_score_slice(z_fixed, q_tokens, y_tokens)
                amax = int(np.argmax(scores))
                token_str = ','.join(str(t) for t in q_tokens)
                ax.plot(t_range, scores, color=color, linewidth=1.5, alpha=0.85,
                        label=f'$x_{{\\min}} = {x_base}$ (argmax $t = {amax}$)')
                ax.axvline(amax, color=color, linewidth=0.6, linestyle=':', alpha=0.4)
            ax.axvline(z_fixed, color='gray', linewidth=1, linestyle='--', alpha=0.5)
            ax.set_xlabel('Key token value ($t$)')
            ax.set_ylabel('Attn2 QK Score')
            ax.set_title(f'$\\delta = {delta}$, $n = {n_tok}$ tokens')
            ax.legend(fontsize=8)

        fig.suptitle(f'Attn2 QK Score Slices: Attn1 Split Equally on {n_tok} Tokens\n'
                     f'($z = {z_fixed}$, tokens: $x, x+\\delta, \\ldots, x+{n_tok-1}\\delta$)',
                     fontsize=14, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.90])
        out = os.path.join(OUTDIR, f'qk_slices_split_{n_tok}tokens.png')
        fig.savefig(out, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  Saved {os.path.basename(out)}")


# ==========================================================================
# HEATMAP PLOTS for n_tokens = 1, 2, 3, 4 with delta=5
# ==========================================================================
def generate_heatmap_plots():
    print("=== Generating heatmap plots ===")
    z_fixed = 250
    y_tokens = [250]
    delta = 5

    fig, axes = plt.subplots(1, 4, figsize=(24, 5.5))
    for ax, n_tok in zip(axes, [1, 2, 3, 4]):
        print(f"  Computing heatmap for n_tokens={n_tok}, delta={delta} ...")
        if n_tok == 1:
            hm = compute_heatmap(z_fixed, y_tokens)
            n_rows = vocab_n
        else:
            hm = compute_heatmap_split(z_fixed, y_tokens, n_tok, delta)
            n_rows = hm.shape[0]

        vmax = np.percentile(np.abs(hm), 99)
        im = ax.imshow(hm, aspect='auto', origin='lower', cmap='RdBu_r',
                       vmin=-vmax, vmax=vmax, interpolation='nearest',
                       extent=[0, vocab_n, 0, n_rows])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xlabel('Key token value ($t$)')
        if n_tok == 1:
            ax.set_ylabel('Query-side attn1 target ($x$)')
            ax.set_title(f'$n = 1$ (single focus)')
        else:
            ax.set_ylabel(f'$x_{{\\min}}$ (smallest of {n_tok} targets)')
            ax.set_title(f'$n = {n_tok}$, $\\delta = {delta}$')

    fig.suptitle(f'Attn2 QK Score Heatmaps: Single vs Split Attn1 ($z = {z_fixed}$)',
                 fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    out = os.path.join(OUTDIR, 'qk_heatmap_split_comparison.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved {os.path.basename(out)}")


# ==========================================================================
# ARGMAX ANALYSIS: Is argmax biased toward x_min?
# ==========================================================================
def argmax_analysis():
    print("\n=== Argmax bias analysis ===")
    z_fixed = 250
    y_tokens = [251]

    print(f"\nQuery token z = {z_fixed}, key-side y = {y_tokens}")
    print(f"For each (n_tokens, delta, x_min), we compute the argmax of the score slice")
    print(f"and compare it to x_min and x_max = x_min + (n-1)*delta.\n")

    header = f"{'n':>3} {'delta':>6} {'x_min':>6} {'x_max':>6} {'x_mean':>7} {'argmax':>7} {'argmax-x_min':>13} {'argmax-x_mean':>14}"
    print(header)
    print("-" * len(header))

    results = []
    for n_tok in [1, 2, 3, 4]:
        for delta in [1, 2, 5, 10, 20]:
            for x_min in [255, 260, 270, 280, 290, 300, 350, 400]:
                q_tokens = [x_min + i * delta for i in range(n_tok)]
                if max(q_tokens) >= vocab_n:
                    continue
                scores = compute_score_slice(z_fixed, q_tokens, y_tokens)
                amax = int(np.argmax(scores))
                x_max = max(q_tokens)
                x_mean = np.mean(q_tokens)
                results.append({
                    'n': n_tok, 'delta': delta, 'x_min': x_min,
                    'x_max': x_max, 'x_mean': x_mean, 'argmax': amax
                })
                print(f"{n_tok:>3} {delta:>6} {x_min:>6} {x_max:>6} {x_mean:>7.1f} {amax:>7} {amax - x_min:>+13} {amax - x_mean:>+14.1f}")

    # Summary statistics
    print("\n\n=== Summary: argmax relative to x_min vs x_mean vs x_max ===\n")
    for n_tok in [1, 2, 3, 4]:
        subset = [r for r in results if r['n'] == n_tok]
        if not subset:
            continue
        diffs_min = [r['argmax'] - r['x_min'] for r in subset]
        diffs_mean = [r['argmax'] - r['x_mean'] for r in subset]
        diffs_max = [r['argmax'] - r['x_max'] for r in subset]
        print(f"n_tokens = {n_tok}:")
        print(f"  argmax - x_min:  mean = {np.mean(diffs_min):+.1f}, median = {np.median(diffs_min):+.1f}")
        print(f"  argmax - x_mean: mean = {np.mean(diffs_mean):+.1f}, median = {np.median(diffs_mean):+.1f}")
        print(f"  argmax - x_max:  mean = {np.mean(diffs_max):+.1f}, median = {np.median(diffs_max):+.1f}")
        print()

    # Detailed comparison: same x_min, varying n_tok and delta
    print("\n=== Detailed: Fixed x_min, how does argmax change with n and delta? ===\n")
    for x_min in [260, 280, 300]:
        print(f"x_min = {x_min}:")
        print(f"  {'n':>3} {'delta':>6} {'tokens':>30} {'argmax':>7}")
        for n_tok in [1, 2, 3, 4]:
            for delta in [1, 5, 10, 20]:
                q_tokens = [x_min + i * delta for i in range(n_tok)]
                if max(q_tokens) >= vocab_n:
                    continue
                scores = compute_score_slice(z_fixed, q_tokens, y_tokens)
                amax = int(np.argmax(scores))
                tok_str = str(q_tokens)
                print(f"  {n_tok:>3} {delta:>6} {tok_str:>30} {amax:>7}")
        print()


if __name__ == '__main__':
    generate_slice_plots()
    generate_heatmap_plots()
    argmax_analysis()
    print("\nDone.")
