#!/usr/bin/env python3
"""
Extract horizontal slices from the (x, t) QK heatmaps at y-axis values
that are ~200 above the query base token z for each subplot.
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
        os.path.dirname(ARGS.ckpt), "..", "plots", "qk_heatmap_xt_slices3.png"
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

    def compute_all_Q_vary_x(z):
        inp = e_all[z].unsqueeze(0) + V_all
        mlp_out = block0.mlp(block0.ln_2(inp))
        h = block1.ln_1(mlp_out)
        return h @ W_q.T + b_q

    def compute_all_K_vary_t(y):
        inp = e_all + V_all[y].unsqueeze(0)
        mlp_out = block0.mlp(block0.ln_2(inp))
        h = block1.ln_1(mlp_out)
        return h @ W_k.T + b_k

    def compute_heatmap_xt(z, y):
        Q = compute_all_Q_vary_x(z)
        K = compute_all_K_vary_t(y)
        return (Q @ K.T).cpu().numpy()

    configs = [
        (250, 250, "z=250, y=250"),
        (250, 251, "z=250, y=251"),
        (251, 250, "z=251, y=250"),
        (100, 100, "z=100, y=100"),
        (100, 101, "z=100, y=101"),
        (400, 401, "z=400, y=401"),
    ]

    slice_colors = ["#1b7837", "#2166ac", "#d6604d", "#762a83", "#e08214"]

    n_cols = 3
    n_rows = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 5.5 * n_rows))
    axes = axes.flatten()
    t_range = np.arange(vocab_n)

    for ci, (z, y, label) in enumerate(configs):
        center = z + 200
        x_slices = [center - 20, center - 10, center, center + 10, center + 20]
        x_slices = [x for x in x_slices if 0 <= x < vocab_n]

        print(f"  {label}: slices at x = {x_slices}")
        hm = compute_heatmap_xt(z, y)

        ax = axes[ci]
        for si, x_val in enumerate(x_slices):
            scores = hm[x_val, :]
            ax.plot(t_range, scores, color=slice_colors[si], linewidth=1.3,
                    label=f"x={x_val}", alpha=0.85)

        ax.set_xlabel("t (key base token)", fontsize=11)
        ax.set_ylabel("QK score", fontsize=11)
        ax.set_title(f"{label}  |  slices at x ≈ z+200 = {center}",
                     fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.15)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        for si, x_val in enumerate(x_slices):
            scores = hm[x_val, :]
            amax = int(np.argmax(scores))
            ymin, ymax = ax.get_ylim()
            ax.axvline(amax, color=slice_colors[si], linewidth=0.5,
                       linestyle=":", alpha=0.35)
            ax.text(amax, scores[amax] + (ymax - ymin) * 0.015,
                    f"{amax}", fontsize=8, color=slice_colors[si],
                    ha="center", va="bottom", fontweight="bold")

    fig.suptitle(
        f"QK score slices at x ≈ z + 200 (query L1 target far from base token)\n"
        f"score(x, t) vs t\n"
        f"k={block_size}, N={vocab_n}, {os.path.basename(ARGS.ckpt)}",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
