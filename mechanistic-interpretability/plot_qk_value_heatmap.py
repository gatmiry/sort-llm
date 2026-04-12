#!/usr/bin/env python3
"""
Compute and visualize the QK interaction heatmap where:

  Q_x = W_Q_L2 @ LN_L2( MLP1( LN_2( V_x ) ) )
  K_y = W_K_L2 @ LN_L2( MLP1( LN_2( V_y ) ) )

  V_x = c_proj_L1( W_v_L1 @ LN_1( e_x ) + b_v )

  score(x, y) = Q_x · K_y

for all pairs (x, y) in the vocabulary.

V_x represents the L1 attention output when fully focused on token x.
The MLP then processes this, and L2's Q/K weights are applied.
"""

import os, sys, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "sort-llm", "sortgpt_toolkit"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from model import DEVICE, load_model_from_checkpoint

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", required=True)
parser.add_argument("--output", default=None)
ARGS = parser.parse_args()


@torch.no_grad()
def main():
    model = load_model_from_checkpoint(ARGS.ckpt)
    n_embd = model.config.n_embd
    vocab_n = model.config.vocab_size - 1
    block_size = model.config.block_size

    block0 = model.transformer.h[0]
    block1 = model.transformer.h[1]

    output_path = ARGS.output or os.path.join(
        os.path.dirname(ARGS.ckpt), "..", "plots", "qk_value_heatmap.png"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Token embeddings (no positional embedding — position-independent)
    e = model.transformer.wte.weight[:vocab_n]  # (N, C)

    # L1 value vectors: V_x = c_proj( W_v @ LN_1(e_x) + b_v )
    ln1_e = block0.ln_1(e)  # (N, C)

    W_v = block0.attn.c_attn.weight[2*n_embd:3*n_embd, :]
    b_v = block0.attn.c_attn.bias[2*n_embd:3*n_embd]
    v = ln1_e @ W_v.T + b_v  # (N, C)

    V = v @ block0.attn.c_proj.weight.T + block0.attn.c_proj.bias  # (N, C)

    # MLP1 output: MLP( LN_2( V_x ) )
    mlp_input = block0.ln_2(V)
    mlp_out = block0.mlp(mlp_input)  # (N, C)

    # L2 Q and K: W_Q/K @ LN_L2(mlp_out) + bias
    h = block1.ln_1(mlp_out)  # (N, C)

    W_q = block1.attn.c_attn.weight[:n_embd, :]
    b_q = block1.attn.c_attn.bias[:n_embd]
    W_k = block1.attn.c_attn.weight[n_embd:2*n_embd, :]
    b_k = block1.attn.c_attn.bias[n_embd:2*n_embd]

    Q = h @ W_q.T + b_q  # (N, C)
    K = h @ W_k.T + b_k  # (N, C)

    # Heatmap: score(x, y) = Q_x · K_y
    heatmap = (Q @ K.T).cpu().numpy()  # (N, N)

    # Self-score: score(x, x)
    self_score = np.diag(heatmap)
    # Neighbor score: score(x, x+1)
    neighbor_score = np.array([heatmap[i, i+1] for i in range(vocab_n - 1)])
    # score(x, x-1)
    prev_score = np.array([heatmap[i, i-1] for i in range(1, vocab_n)])

    # ========== PLOTTING ==========
    fig = plt.figure(figsize=(28, 20))
    gs = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.35,
                          height_ratios=[1.2, 1, 1])

    # (0,0:2): Full heatmap
    ax = fig.add_subplot(gs[0, 0:2])
    vmax = np.percentile(np.abs(heatmap), 99)
    im = ax.imshow(heatmap, aspect="auto", origin="lower",
                   cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   interpolation="nearest")
    ax.set_xlabel("Key token y", fontsize=11)
    ax.set_ylabel("Query token x", fontsize=11)
    ax.set_title("Full QK heatmap: score(x, y)\nQ_x = W_Q @ LN(MLP(LN(V_x))),  K_y = W_K @ LN(MLP(LN(V_y)))",
                 fontsize=11, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # (0,2:4): Zoomed-in region (e.g., 200-280)
    ax = fig.add_subplot(gs[0, 2:4])
    lo, hi = 200, 280
    hm_zoom = heatmap[lo:hi, lo:hi]
    vmax_z = np.percentile(np.abs(hm_zoom), 99)
    im = ax.imshow(hm_zoom, aspect="auto", origin="lower",
                   cmap="RdBu_r", vmin=-vmax_z, vmax=vmax_z,
                   extent=[lo, hi, lo, hi], interpolation="nearest")
    ax.set_xlabel("Key token y", fontsize=11)
    ax.set_ylabel("Query token x", fontsize=11)
    ax.set_title(f"Zoomed: tokens {lo}–{hi}", fontsize=11, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # (1,0): Self-score and neighbor-score
    ax = fig.add_subplot(gs[1, 0:2])
    ax.plot(range(vocab_n), self_score, linewidth=0.8, color="#2d2d2d", label="Self: score(x, x)")
    ax.plot(range(vocab_n - 1), neighbor_score, linewidth=0.8, color="#b2182b", label="Next: score(x, x+1)")
    ax.plot(range(1, vocab_n), prev_score, linewidth=0.8, color="#2166ac", label="Prev: score(x, x−1)")
    ax.set_xlabel("Token x", fontsize=11)
    ax.set_ylabel("QK score", fontsize=11)
    ax.set_title("Self vs neighbor scores", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # (1,2): Difference: score(x,x+1) - score(x,x)
    ax = fig.add_subplot(gs[1, 2])
    diff_next = neighbor_score - self_score[:-1]
    n_pos = np.sum(diff_next > 0)
    n_neg = np.sum(diff_next < 0)
    ax.plot(range(vocab_n - 1), diff_next, linewidth=0.8, color="#b2182b")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Token x", fontsize=11)
    ax.set_ylabel("score(x,x+1) − score(x,x)", fontsize=10)
    ax.set_title(f"Next − Self  (+:{n_pos}, −:{n_neg})", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # (1,3): Difference: score(x,x+1) - score(x,x-1)
    ax = fig.add_subplot(gs[1, 3])
    diff_asym = neighbor_score[1:] - prev_score[:-1]  # for x=1..N-2
    n_pos = np.sum(diff_asym > 0)
    n_neg = np.sum(diff_asym < 0)
    ax.plot(range(1, vocab_n - 1), diff_asym, linewidth=0.8, color="#762a83")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Token x", fontsize=11)
    ax.set_ylabel("score(x,x+1) − score(x,x−1)", fontsize=10)
    ax.set_title(f"Next − Prev asymmetry  (+:{n_pos}, −:{n_neg})", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # (2,0): Local profile: for a few query tokens, show score(x, y) for y near x
    ax = fig.add_subplot(gs[2, 0:2])
    window = 30
    sample_xs = [50, 150, 250, 350, 450]
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(sample_xs)))
    for x_val, col in zip(sample_xs, colors):
        lo_y = max(0, x_val - window)
        hi_y = min(vocab_n, x_val + window + 1)
        ys = np.arange(lo_y, hi_y)
        scores = heatmap[x_val, lo_y:hi_y]
        ax.plot(ys - x_val, scores, linewidth=1.2, color=col, label=f"x={x_val}")
    ax.axvline(0, color="black", linewidth=0.5, linestyle=":")
    ax.axvline(1, color="red", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.set_xlabel("y − x (offset from query)", fontsize=11)
    ax.set_ylabel("score(x, y)", fontsize=11)
    ax.set_title("Local QK profile around query x", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # (2,2): Argmax analysis: for each x, which y has highest score?
    ax = fig.add_subplot(gs[2, 2])
    argmax_y = np.argmax(heatmap, axis=1)
    offsets = argmax_y - np.arange(vocab_n)
    ax.scatter(range(vocab_n), offsets, s=1, alpha=0.5, color="#2d2d2d")
    ax.set_xlabel("Query token x", fontsize=11)
    ax.set_ylabel("argmax_y score(x,y) − x", fontsize=10)
    ax.set_title("Best-matching key offset\n(argmax_y − x)", fontsize=11, fontweight="bold")
    # Count how many have offset = +1
    n_plus1 = np.sum(offsets == 1)
    n_zero = np.sum(offsets == 0)
    ax.axhline(1, color="red", linewidth=1, linestyle="--", alpha=0.5, label=f"+1: {n_plus1}/{vocab_n}")
    ax.axhline(0, color="blue", linewidth=1, linestyle="--", alpha=0.5, label=f"0: {n_zero}/{vocab_n}")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # (2,3): Distribution of argmax offsets
    ax = fig.add_subplot(gs[2, 3])
    offset_counts = {}
    for o in offsets:
        offset_counts[o] = offset_counts.get(o, 0) + 1
    sorted_offsets = sorted(offset_counts.keys())
    # Focus on offsets within ±10
    focus_offsets = [o for o in sorted_offsets if -10 <= o <= 10]
    counts_fo = [offset_counts[o] for o in focus_offsets]
    bars = ax.bar(focus_offsets, counts_fo, color="#2d2d2d", alpha=0.75, edgecolor="black", linewidth=0.3)
    for o_val in focus_offsets:
        if offset_counts[o_val] > 10:
            ax.text(o_val, offset_counts[o_val] + 2,
                    str(offset_counts[o_val]), ha="center", fontsize=7, fontweight="bold")
    ax.set_xlabel("Argmax offset (best_y − x)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Distribution of best-matching\nkey offset", fontsize=11, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle(
        f"QK heatmap: L1 value vectors through MLP1 → L2 Q/K\n"
        f"k={block_size}, N={vocab_n}, {os.path.basename(ARGS.ckpt)}",
        fontsize=14, fontweight="bold",
    )
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")

    # Summary stats
    print(f"\n=== Summary ===")
    print(f"  Mean self-score score(x,x):     {self_score.mean():.1f}")
    print(f"  Mean next-score score(x,x+1):   {neighbor_score.mean():.1f}")
    print(f"  Mean prev-score score(x,x-1):   {prev_score.mean():.1f}")
    print(f"  score(x,x+1) > score(x,x):     {np.sum(neighbor_score > self_score[:-1])}/{vocab_n-1}")
    print(f"  score(x,x+1) > score(x,x-1):   {np.sum(neighbor_score[1:] > prev_score[:-1])}/{vocab_n-2}")
    print(f"  Argmax offset = +1:             {n_plus1}/{vocab_n} ({n_plus1/vocab_n:.1%})")
    print(f"  Argmax offset = 0:              {n_zero}/{vocab_n} ({n_zero/vocab_n:.1%})")


if __name__ == "__main__":
    main()
