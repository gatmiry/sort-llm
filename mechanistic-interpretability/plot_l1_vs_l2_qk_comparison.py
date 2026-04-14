#!/usr/bin/env python3
"""
Compare L1 QK interaction (query=250 vs all keys) with L2 QK score slices
(from qk_heatmap_xt, z=250, y=250, x=260..300 vs all t) on the same x-axis.
Two y-axes since the scales differ.
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
        os.path.dirname(ARGS.ckpt), "..", "plots", "l1_vs_l2_qk_comparison.png"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    e_all = model.transformer.wte.weight[:vocab_n]

    # --- L1 QK scores for query=250 ---
    W_q_L1 = block0.attn.c_attn.weight[:n_embd, :]
    b_q_L1 = block0.attn.c_attn.bias[:n_embd]
    W_k_L1 = block0.attn.c_attn.weight[n_embd:2*n_embd, :]
    b_k_L1 = block0.attn.c_attn.bias[n_embd:2*n_embd]
    ln1_L0 = block0.ln_1

    h_q_l1 = ln1_L0(e_all[250].unsqueeze(0))
    Q_l1 = h_q_l1 @ W_q_L1.T + b_q_L1
    h_k_l1 = ln1_L0(e_all)
    K_l1 = h_k_l1 @ W_k_L1.T + b_k_L1
    l1_scores = (Q_l1 @ K_l1.T).squeeze(0).cpu().numpy()

    # --- L2 QK scores: z=250, y=250, slices at x=260..300 ---
    ln1_L0_for_v = block0.ln_1
    W_v = block0.attn.c_attn.weight[2*n_embd:3*n_embd, :]
    b_v = block0.attn.c_attn.bias[2*n_embd:3*n_embd]
    v = ln1_L0_for_v(e_all) @ W_v.T + b_v
    V_all = v @ block0.attn.c_proj.weight.T + block0.attn.c_proj.bias

    W_q_L2 = block1.attn.c_attn.weight[:n_embd, :]
    b_q_L2 = block1.attn.c_attn.bias[:n_embd]
    W_k_L2 = block1.attn.c_attn.weight[n_embd:2*n_embd, :]
    b_k_L2 = block1.attn.c_attn.bias[n_embd:2*n_embd]

    z, y = 250, 250
    inp_q = e_all[z].unsqueeze(0) + V_all
    mlp_q = block0.mlp(block0.ln_2(inp_q))
    Q_l2_all = block1.ln_1(mlp_q) @ W_q_L2.T + b_q_L2  # (N, C)

    inp_k = e_all + V_all[y].unsqueeze(0)
    mlp_k = block0.mlp(block0.ln_2(inp_k))
    K_l2_all = block1.ln_1(mlp_k) @ W_k_L2.T + b_k_L2  # (N, C)

    hm = (Q_l2_all @ K_l2_all.T).cpu().numpy()  # (N, N)

    x_slices = [260, 270, 280, 290, 300]
    slice_colors = ["#1b7837", "#2166ac", "#d6604d", "#762a83", "#e08214"]
    t_range = np.arange(vocab_n)

    fig, ax1 = plt.subplots(figsize=(16, 7))

    # L1 on primary y-axis
    ax1.plot(t_range, l1_scores, color="black", linewidth=2.0, alpha=0.7,
             label="L1: query=250 vs all keys", zorder=10)
    ax1.set_xlabel("Token value (key side)", fontsize=12)
    ax1.set_ylabel("L1 QK score (pre-softmax)", fontsize=12, color="black")
    ax1.tick_params(axis="y", labelcolor="black")

    # L2 slices on secondary y-axis
    ax2 = ax1.twinx()
    for si, x_val in enumerate(x_slices):
        scores = hm[x_val, :]
        ax2.plot(t_range, scores, color=slice_colors[si], linewidth=1.2,
                 label=f"L2: x={x_val}, z=250, y=250", alpha=0.8)
    ax2.set_ylabel("L2 QK score", fontsize=12, color="#2166ac")
    ax2.tick_params(axis="y", labelcolor="#2166ac")

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper left")

    ax1.axvline(250, color="red", linewidth=0.8, linestyle="--", alpha=0.4,
                label="token 250")
    ax1.grid(True, alpha=0.15)
    ax1.set_title(
        f"L1 QK interaction (query=250) vs L2 QK score slices (z=250, y=250)\n"
        f"k={block_size}, N={vocab_n}, {os.path.basename(ARGS.ckpt)}",
        fontsize=13, fontweight="bold",
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
