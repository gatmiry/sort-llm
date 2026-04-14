#!/usr/bin/env python3
"""
Heatmap of QK score varying (x, t) where L1 attention is split 50/50
between two numbers x and x+d (instead of fully focused on x).

  V_split(x) = 0.5 * V_x + 0.5 * V_{x+d}

  Q(x) = W_Q_L2 @ LN_L2( MLP1( LN_2( e_z + V_split(x) ) ) )
  K(t) = W_K_L2 @ LN_L2( MLP1( LN_2( e_t + V_split(y) ) ) )

  score(x, t) = Q(x) · K(t)

y-axis = x (smaller of the two numbers L1 attends to on query side)
x-axis = t (key base token)
Multiple subplots for different fixed gap distances d.
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
        os.path.dirname(ARGS.ckpt), "..", "plots", "qk_heatmap_xt_split.png"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    e_all = model.transformer.wte.weight[:vocab_n]  # (N, C)
    ln1_e = block0.ln_1(e_all)
    W_v = block0.attn.c_attn.weight[2*n_embd:3*n_embd, :]
    b_v = block0.attn.c_attn.bias[2*n_embd:3*n_embd]
    v = ln1_e @ W_v.T + b_v
    V_all = v @ block0.attn.c_proj.weight.T + block0.attn.c_proj.bias  # (N, C)

    W_q = block1.attn.c_attn.weight[:n_embd, :]
    b_q = block1.attn.c_attn.bias[:n_embd]
    W_k = block1.attn.c_attn.weight[n_embd:2*n_embd, :]
    b_k = block1.attn.c_attn.bias[n_embd:2*n_embd]

    def compute_Q_split(z, d):
        """Q(x) for all valid x, with L1 attn split 50/50 on (x, x+d).

        Returns (N-d, C) for x in [0, vocab_n - d).
        """
        n_valid = vocab_n - d
        V_split = 0.5 * V_all[:n_valid] + 0.5 * V_all[d:d + n_valid]  # (N-d, C)
        inp = e_all[z].unsqueeze(0) + V_split
        mlp_out = block0.mlp(block0.ln_2(inp))
        h = block1.ln_1(mlp_out)
        return h @ W_q.T + b_q  # (N-d, C)

    def compute_K_split(y, d):
        """K(t) for all t, with L1 attn split 50/50 on (y, y+d).

        Returns (N, C).
        """
        V_split_y = 0.5 * V_all[y] + 0.5 * V_all[y + d]  # (C,)
        inp = e_all + V_split_y.unsqueeze(0)
        mlp_out = block0.mlp(block0.ln_2(inp))
        h = block1.ln_1(mlp_out)
        return h @ W_k.T + b_k  # (N, C)

    def compute_heatmap(z, y, d):
        """score(x, t) with split attention, gap d."""
        Q = compute_Q_split(z, d)  # (N-d, C)
        K = compute_K_split(y, d)  # (N, C)
        return (Q @ K.T).cpu().numpy()  # (N-d, N)

    # Fixed (z, y) pair; vary d
    z_fixed, y_fixed = 250, 251
    gaps = [1, 2, 5, 10, 20, 50]

    n_cols = 3
    n_rows = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7.5 * n_cols, 6.5 * n_rows))
    axes = axes.flatten()

    for idx, d in enumerate(gaps):
        if y_fixed + d >= vocab_n:
            continue
        print(f"  Computing heatmap for gap d={d} ...")
        hm = compute_heatmap(z_fixed, y_fixed, d)
        n_valid = vocab_n - d

        ax = axes[idx]
        vmax = np.percentile(np.abs(hm), 99)
        if vmax < 1:
            vmax = 1
        im = ax.imshow(hm, aspect="auto", origin="lower",
                       cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                       interpolation="nearest",
                       extent=[0, vocab_n, 0, n_valid])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, format="%.0f")

        ax.set_xlabel("t (key base token)", fontsize=10)
        ax.set_ylabel("x (smaller L1 target, attends x & x+d)", fontsize=9)
        ax.set_title(f"d = {d}   (z={z_fixed}, y={y_fixed})\nL1 attn: 0.5·V_x + 0.5·V_{{x+{d}}}",
                     fontsize=10, fontweight="bold")

        ax.plot([0, min(vocab_n, n_valid)], [0, min(vocab_n, n_valid)],
                "w--", linewidth=0.5, alpha=0.5)

        # For reference: where is x such that x+d = t (i.e. the "next value" diagonal)?
        # If the model matches x+d+1 on the key side, diagonal would be at t = x+d+1
        # Mark the +d offset diagonal
        ax.plot([d, min(vocab_n, n_valid + d)], [0, min(n_valid, vocab_n - d)],
                "c--", linewidth=0.7, alpha=0.5, label=f"t = x+{d}")
        if d > 1:
            ax.plot([d + 1, min(vocab_n, n_valid + d + 1)], [0, min(n_valid, vocab_n - d - 1)],
                    "lime", linewidth=0.7, linestyle="--", alpha=0.5, label=f"t = x+{d}+1")

        argmax_t = np.argmax(hm, axis=1)
        n_match_x1 = np.sum(argmax_t == np.arange(n_valid) + 1)
        n_match_xd1 = np.sum(argmax_t == np.arange(n_valid) + d + 1)
        median_offset = np.median(argmax_t - np.arange(n_valid))

        ann = (f"argmax_t=x+1: {n_match_x1}/{n_valid}\n"
               f"argmax_t=x+d+1: {n_match_xd1}/{n_valid}\n"
               f"median offset: {median_offset:.0f}")
        ax.text(0.02, 0.98, ann, transform=ax.transAxes, fontsize=7,
                verticalalignment="top", fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))
        if d > 1:
            ax.legend(fontsize=7, loc="lower right")

    fig.suptitle(
        f"QK heatmap with split L1 attention: 0.5·V_x + 0.5·V_{{x+d}}\n"
        f"Q(x) = W_Q·LN(MLP(LN(e_z + V_split))),  K(t) = W_K·LN(MLP(LN(e_t + V_split)))\n"
        f"z={z_fixed}, y={y_fixed},  y-axis = x (smaller target),  x-axis = t (key base)\n"
        f"k={block_size}, N={vocab_n}, {os.path.basename(ARGS.ckpt)}",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
