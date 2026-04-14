#!/usr/bin/env python3
"""
Extract horizontal slices from the split-attention (x, t) QK heatmaps
at x = 260, 270, 280, 290, 300, for each gap distance d.
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
        os.path.dirname(ARGS.ckpt), "..", "plots", "qk_heatmap_xt_split_slices.png"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

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

    z_fixed, y_fixed = 250, 251
    gaps = [1, 2, 5, 10, 20, 50]
    x_slices = [260, 270, 280, 290, 300]
    slice_colors = ["#1b7837", "#2166ac", "#d6604d", "#762a83", "#e08214"]

    def compute_Q_split(z, d):
        n_valid = vocab_n - d
        V_split = 0.5 * V_all[:n_valid] + 0.5 * V_all[d:d + n_valid]
        inp = e_all[z].unsqueeze(0) + V_split
        mlp_out = block0.mlp(block0.ln_2(inp))
        h = block1.ln_1(mlp_out)
        return h @ W_q.T + b_q

    def compute_K_split(y, d):
        V_split_y = 0.5 * V_all[y] + 0.5 * V_all[y + d]
        inp = e_all + V_split_y.unsqueeze(0)
        mlp_out = block0.mlp(block0.ln_2(inp))
        h = block1.ln_1(mlp_out)
        return h @ W_k.T + b_k

    def compute_heatmap(z, y, d):
        Q = compute_Q_split(z, d)
        K = compute_K_split(y, d)
        return (Q @ K.T).cpu().numpy()

    n_cols = 3
    n_rows = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 5.5 * n_rows))
    axes = axes.flatten()
    t_range = np.arange(vocab_n)

    for gi, d in enumerate(gaps):
        if y_fixed + d >= vocab_n:
            continue
        n_valid = vocab_n - d
        print(f"  Computing heatmap for d={d} ...")
        hm = compute_heatmap(z_fixed, y_fixed, d)

        ax = axes[gi]
        for si, x_val in enumerate(x_slices):
            if x_val >= n_valid:
                continue
            scores = hm[x_val, :]
            ax.plot(t_range, scores, color=slice_colors[si], linewidth=1.3,
                    label=f"x={x_val} (attends {x_val}&{x_val+d})", alpha=0.85)

        ax.set_xlabel("t (key base token)", fontsize=11)
        ax.set_ylabel("QK score", fontsize=11)
        ax.set_title(f"d = {d}   (z={z_fixed}, y={y_fixed})\n"
                     f"L1 attn: 0.5·V_x + 0.5·V_{{x+{d}}}",
                     fontsize=11, fontweight="bold")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.15)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        for si, x_val in enumerate(x_slices):
            if x_val >= n_valid:
                continue
            scores = hm[x_val, :]
            amax = int(np.argmax(scores))
            ymin, ymax = ax.get_ylim()
            ax.axvline(amax, color=slice_colors[si], linewidth=0.5,
                       linestyle=":", alpha=0.35)
            ax.text(amax, scores[amax] + (ymax - ymin) * 0.015,
                    f"{amax}", fontsize=8, color=slice_colors[si],
                    ha="center", va="bottom", fontweight="bold")

    fig.suptitle(
        f"QK score slices from split-attention heatmaps at x = {x_slices}\n"
        f"score(x, t) vs t,  z={z_fixed}, y={y_fixed}\n"
        f"k={block_size}, N={vocab_n}, {os.path.basename(ARGS.ckpt)}",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
