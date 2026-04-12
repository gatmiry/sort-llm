#!/usr/bin/env python3
"""
Mean + individual traces of (W_Q v_i)^T (W_K w_j) as a function of j-i.

w_j = LN_block1(e_j)
v_i: (a) no MLP, (b) with MLP

Top row: all individual traces with bold mean line.
Bottom row: mean ± std shaded region.
Also split by low/mid/high query token range.
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
ARGS = parser.parse_args()


@torch.no_grad()
def main():
    print(f"Loading {ARGS.ckpt}")
    model = load_model_from_checkpoint(ARGS.ckpt)
    n_embd = model.config.n_embd
    vocab_n = model.config.vocab_size - 1
    block_size = model.config.block_size

    output_path = ARGS.output or os.path.join(
        os.path.dirname(ARGS.ckpt), "..", "plots", "qk_cross_mean.png"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    block0 = model.transformer.h[0]
    block1 = model.transformer.h[1]

    W_q2 = block1.attn.c_attn.weight[:n_embd, :]
    W_k2 = block1.attn.c_attn.weight[n_embd:2*n_embd, :]

    e = model.transformer.wte.weight[:vocab_n]

    w = block1.ln_1(e)
    K_w = w @ W_k2.T

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

    # Full score matrices
    score_no_mlp = (Q_no_mlp @ K_w.T).cpu().numpy()
    score_with_mlp = (Q_with_mlp @ K_w.T).cpu().numpy()

    W = ARGS.window
    margin = W + 2
    valid_range = range(margin, vocab_n - margin)
    offsets = np.arange(-W, W + 1)

    def collect_curves(score_mat, tokens):
        curves = []
        for qi in tokens:
            row = [score_mat[qi, qi + off] for off in offsets]
            curves.append(row)
        return np.array(curves)

    # Split into thirds
    n = len(valid_range)
    all_tokens = list(valid_range)
    thirds = [
        ("Low tokens", all_tokens[:n//3]),
        ("Mid tokens", all_tokens[n//3:2*n//3]),
        ("High tokens", all_tokens[2*n//3:]),
    ]
    third_colors = ["#1b7837", "#762a83", "#c51b7d"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    for col, (score_mat, v_label) in enumerate([
        (score_no_mlp, "v without MLP"),
        (score_with_mlp, "v with MLP"),
    ]):
        # Top: individual traces + bold mean
        ax = axes[0][col]
        all_curves = collect_curves(score_mat, all_tokens)
        for row in all_curves[::5]:  # plot every 5th to reduce clutter
            ax.plot(offsets, row, alpha=0.08, linewidth=0.5, color="steelblue")
        mean_curve = np.mean(all_curves, axis=0)
        ax.plot(offsets, mean_curve, color="black", linewidth=2.5, label="Mean (all)")
        ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.7, alpha=0.6)
        ax.axvline(x=1, color="red", linestyle="--", linewidth=0.5, alpha=0.5)
        ax.set_title(f"Individual traces + mean — {v_label}", fontweight="bold")
        ax.set_xlabel("j − i")
        ax.set_ylabel("Score")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.15)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Bottom: mean ± std for each third
        ax = axes[1][col]
        for (label, tokens), color in zip(thirds, third_colors):
            curves = collect_curves(score_mat, tokens)
            mu = np.mean(curves, axis=0)
            std = np.std(curves, axis=0)
            ax.plot(offsets, mu, color=color, linewidth=2, label=label)
            ax.fill_between(offsets, mu - std, mu + std, color=color, alpha=0.12)
        ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.7, alpha=0.6)
        ax.axvline(x=1, color="red", linestyle="--", linewidth=0.5, alpha=0.5)
        ax.set_title(f"Mean ± std by vocab region — {v_label}", fontweight="bold")
        ax.set_xlabel("j − i")
        ax.set_ylabel("Score")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.15)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        f"$(W_Q \\, v_i)^\\top (W_K \\, w_j)$  local around $i$  |  "
        f"$w_j = LN(e_j)$\n"
        f"k={block_size}, N={vocab_n}, {os.path.basename(ARGS.ckpt)}  "
        f"(gray=i, red=i+1)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
