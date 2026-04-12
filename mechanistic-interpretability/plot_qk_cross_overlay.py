#!/usr/bin/env python3
"""
Overlay line plot of (W_Q v_i)^T (W_K w_j) as a function of offset j-i.

w_j = LN_block1(e_j)
v_i: version a (no MLP), version b (with MLP)

All query tokens overlaid on the same axes so shape is easy to compare.
"""

import os
import sys
import argparse

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
parser.add_argument("--window", type=int, default=25)
parser.add_argument("--n-queries", type=int, default=30, help="Number of evenly-spaced query tokens")
ARGS = parser.parse_args()


@torch.no_grad()
def main():
    print(f"Loading {ARGS.ckpt}")
    model = load_model_from_checkpoint(ARGS.ckpt)
    n_embd = model.config.n_embd
    vocab_n = model.config.vocab_size - 1
    block_size = model.config.block_size

    output_path = ARGS.output or os.path.join(
        os.path.dirname(ARGS.ckpt), "..", "plots", "qk_cross_overlay.png"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    block0 = model.transformer.h[0]
    block1 = model.transformer.h[1]

    W_q2 = block1.attn.c_attn.weight[:n_embd, :]
    W_k2 = block1.attn.c_attn.weight[n_embd:2*n_embd, :]

    e = model.transformer.wte.weight[:vocab_n]

    # w_j: raw embedding through LN before Layer 2
    w = block1.ln_1(e)
    K_w = w @ W_k2.T  # (vocab_n, n_embd)

    # v_i without MLP
    e_3d = e.unsqueeze(1)
    h = block0.ln_1(e_3d)
    attn_out = block0.attn(h)
    x_no_mlp = e_3d + attn_out
    v_no_mlp = block1.ln_1(x_no_mlp).squeeze(1)
    Q_no_mlp = v_no_mlp @ W_q2.T

    # v_i with MLP
    x_with_mlp = x_no_mlp.clone()
    x_with_mlp = x_with_mlp + block0.mlp(block0.ln_2(x_with_mlp))
    v_with_mlp = block1.ln_1(x_with_mlp).squeeze(1)
    Q_with_mlp = v_with_mlp @ W_q2.T

    # Full score matrices: (vocab_n, vocab_n)  score[i,j] = Q[i] . K[j]
    score_no_mlp = (Q_no_mlp @ K_w.T).cpu().numpy()
    score_with_mlp = (Q_with_mlp @ K_w.T).cpu().numpy()

    W = ARGS.window
    margin = W + 5
    step = max(1, (vocab_n - 2 * margin) // ARGS.n_queries)
    query_tokens = list(range(margin, vocab_n - margin, step))[:ARGS.n_queries]

    offsets = np.arange(-W, W + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    cmap = plt.cm.viridis
    colors = [cmap(x) for x in np.linspace(0, 1, len(query_tokens))]

    for col, (score_mat, v_label) in enumerate([
        (score_no_mlp, "v without MLP"),
        (score_with_mlp, "v with MLP"),
    ]):
        # Row 0: raw scores overlaid
        ax = axes[0][col]
        for idx, qi in enumerate(query_tokens):
            vals = []
            for off in offsets:
                j = qi + off
                if 0 <= j < vocab_n:
                    vals.append(score_mat[qi, j])
                else:
                    vals.append(np.nan)
            ax.plot(offsets, vals, alpha=0.4, linewidth=0.8, color=colors[idx])
        ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.7, alpha=0.6)
        ax.axvline(x=1, color="red", linestyle="--", linewidth=0.5, alpha=0.5)
        ax.set_title(f"Raw scores — {v_label}", fontweight="bold")
        ax.set_xlabel("j − i")
        ax.set_ylabel("Score")
        ax.grid(True, alpha=0.15)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Row 1: scores normalized per query (subtract self-score)
        ax = axes[1][col]
        for idx, qi in enumerate(query_tokens):
            vals = []
            for off in offsets:
                j = qi + off
                if 0 <= j < vocab_n:
                    vals.append(score_mat[qi, j])
                else:
                    vals.append(np.nan)
            vals = np.array(vals)
            self_val = score_mat[qi, qi]
            vals_centered = vals - self_val
            ax.plot(offsets, vals_centered, alpha=0.4, linewidth=0.8, color=colors[idx])
        ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.7, alpha=0.6)
        ax.axvline(x=1, color="red", linestyle="--", linewidth=0.5, alpha=0.5)
        ax.set_title(f"Centered (subtract self-score) — {v_label}", fontweight="bold")
        ax.set_xlabel("j − i")
        ax.set_ylabel("Score − self_score")
        ax.grid(True, alpha=0.15)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(query_tokens[0], query_tokens[-1]))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.6, pad=0.02)
    cbar.set_label("Query token i", fontsize=10)

    fig.suptitle(
        f"$(W_Q \\, v_i)^\\top (W_K \\, w_j)$  local around $i$  |  "
        f"$w_j = LN(e_j)$\n"
        f"k={block_size}, N={vocab_n}, {os.path.basename(ARGS.ckpt)}",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 0.93, 0.92])
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
