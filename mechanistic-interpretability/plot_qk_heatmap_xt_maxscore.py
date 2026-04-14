#!/usr/bin/env python3
"""
For the z=250, y=250 heatmap from qk_heatmap_xt, extract horizontal slices
at x = 250, 255, 260, ..., 500 and plot the maximum score (over all t)
as a function of x.
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
        os.path.dirname(ARGS.ckpt), "..", "plots", "qk_heatmap_xt_maxscore.png"
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

    z, y = 250, 250

    # Compute full heatmap once
    inp_q = e_all[z].unsqueeze(0) + V_all
    mlp_q = block0.mlp(block0.ln_2(inp_q))
    Q_all = block1.ln_1(mlp_q) @ W_q.T + b_q

    inp_k = e_all + V_all[y].unsqueeze(0)
    mlp_k = block0.mlp(block0.ln_2(inp_k))
    K_all = block1.ln_1(mlp_k) @ W_k.T + b_k

    hm = (Q_all @ K_all.T).cpu().numpy()  # (N, N): rows=x, cols=t

    x_values = list(range(250, min(501, vocab_n), 5))
    max_scores = []
    argmax_ts = []
    for x_val in x_values:
        row = hm[x_val, :]
        max_scores.append(np.max(row))
        argmax_ts.append(int(np.argmax(row)))

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # Left: max score vs x
    ax = axes[0]
    ax.plot(x_values, max_scores, "o-", color="#2166ac", linewidth=1.5, markersize=4)
    ax.set_xlabel("x (query L1 target)", fontsize=12)
    ax.set_ylabel("max_t score(x, t)", fontsize=12)
    ax.set_title(f"Maximum QK score over all t\nz={z}, y={y}", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Right: argmax t vs x
    ax = axes[1]
    ax.plot(x_values, argmax_ts, "o-", color="#b2182b", linewidth=1.5, markersize=4)
    ax.axhline(z, color="gray", linewidth=0.8, linestyle="--", alpha=0.5, label=f"z={z}")
    ax.plot(x_values, x_values, "k--", linewidth=0.8, alpha=0.3, label="t = x (diagonal)")
    ax.set_xlabel("x (query L1 target)", fontsize=12)
    ax.set_ylabel("argmax_t score(x, t)", fontsize=12)
    ax.set_title(f"Which key base token t maximizes the score\nz={z}, y={y}",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle(
        f"Max QK score and argmax-t as x varies from 250 to 500 (step 5)\n"
        f"z={z}, y={y},  k={block_size}, N={vocab_n}, {os.path.basename(ARGS.ckpt)}",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")

    print(f"\nx range: {x_values[0]}..{x_values[-1]}")
    print(f"Max score range: {min(max_scores):.1f} .. {max(max_scores):.1f}")
    print(f"Argmax t range: {min(argmax_ts)} .. {max(argmax_ts)}")
    print(f"Mean argmax t: {np.mean(argmax_ts):.1f}")


if __name__ == "__main__":
    main()
