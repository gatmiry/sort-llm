#!/usr/bin/env python3
"""
Plot the Layer 1 QK interaction score between a fixed query token and
all vocabulary tokens as keys.

score(q_token, k_token) = (W_Q @ LN_1(e_{q_token})) · (W_K @ LN_1(e_{k_token}))
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
parser.add_argument("--query-token", type=int, default=250)
ARGS = parser.parse_args()


@torch.no_grad()
def main():
    model = load_model_from_checkpoint(ARGS.ckpt)
    n_embd = model.config.n_embd
    vocab_n = model.config.vocab_size - 1
    block_size = model.config.block_size

    block0 = model.transformer.h[0]

    output_path = ARGS.output or os.path.join(
        os.path.dirname(ARGS.ckpt), "..", "plots", "l1_qk_interaction.png"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    e_all = model.transformer.wte.weight[:vocab_n]  # (N, C)

    W_q = block0.attn.c_attn.weight[:n_embd, :]
    b_q = block0.attn.c_attn.bias[:n_embd]
    W_k = block0.attn.c_attn.weight[n_embd:2*n_embd, :]
    b_k = block0.attn.c_attn.bias[n_embd:2*n_embd]

    ln1 = block0.ln_1

    # Compute Q for the fixed query token
    q_token = ARGS.query_token
    h_q = ln1(e_all[q_token].unsqueeze(0))  # (1, C)
    Q = h_q @ W_q.T + b_q  # (1, C)

    # Compute K for all vocabulary tokens
    h_k = ln1(e_all)  # (N, C)
    K = h_k @ W_k.T + b_k  # (N, C)

    scores = (Q @ K.T).squeeze(0).cpu().numpy()  # (N,)

    argmax_k = int(np.argmax(scores))
    argmin_k = int(np.argmin(scores))

    # Also compute for a few nearby query tokens for comparison
    query_tokens = [q_token - 10, q_token - 5, q_token, q_token + 5, q_token + 10]
    query_tokens = [t for t in query_tokens if 0 <= t < vocab_n]
    q_colors = ["#1b7837", "#2166ac", "#d6604d", "#762a83", "#e08214"]

    fig, axes = plt.subplots(1, 2, figsize=(20, 7))

    # Left: single query token, full range
    ax = axes[0]
    ax.plot(np.arange(vocab_n), scores, color="#2166ac", linewidth=1.0, alpha=0.8)
    ax.axvline(q_token, color="red", linewidth=0.8, linestyle="--", alpha=0.5,
               label=f"query token = {q_token}")
    ax.axvline(argmax_k, color="green", linewidth=0.8, linestyle=":", alpha=0.5)
    ax.text(argmax_k, scores[argmax_k], f"  max: k={argmax_k}", fontsize=9,
            color="green", fontweight="bold", va="bottom")

    ax.set_xlabel("Key token value", fontsize=12)
    ax.set_ylabel("L1 QK score", fontsize=12)
    ax.set_title(f"Layer 1 QK interaction: query token = {q_token}\n"
                 f"score = (W_Q·LN₁(e_q)) · (W_K·LN₁(e_k))",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ann = (f"argmax key: {argmax_k} (score={scores[argmax_k]:.1f})\n"
           f"argmin key: {argmin_k} (score={scores[argmin_k]:.1f})\n"
           f"score at self ({q_token}): {scores[q_token]:.1f}\n"
           f"score at {q_token}+1: {scores[min(q_token+1, vocab_n-1)]:.1f}\n"
           f"score at {q_token}-1: {scores[max(q_token-1, 0)]:.1f}")
    ax.text(0.98, 0.98, ann, transform=ax.transAxes, fontsize=9,
            verticalalignment="top", ha="right", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # Right: multiple query tokens overlaid
    ax = axes[1]
    for qi, qt in enumerate(query_tokens):
        h_qi = ln1(e_all[qt].unsqueeze(0))
        Q_i = h_qi @ W_q.T + b_q
        scores_i = (Q_i @ K.T).squeeze(0).cpu().numpy()
        ax.plot(np.arange(vocab_n), scores_i, color=q_colors[qi], linewidth=1.0,
                label=f"query={qt}", alpha=0.8)

    ax.set_xlabel("Key token value", fontsize=12)
    ax.set_ylabel("L1 QK score", fontsize=12)
    ax.set_title(f"Layer 1 QK interaction: multiple query tokens\naround {q_token}",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle(
        f"Layer 1 attention QK scores (pre-softmax, no positional embedding)\n"
        f"k={block_size}, N={vocab_n}, {os.path.basename(ARGS.ckpt)}",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")

    print(f"\nQuery token: {q_token}")
    print(f"  argmax key: {argmax_k} (score={scores[argmax_k]:.1f})")
    print(f"  score at self: {scores[q_token]:.1f}")
    print(f"  score at +1: {scores[min(q_token+1, vocab_n-1)]:.1f}")
    print(f"  score at -1: {scores[max(q_token-1, 0)]:.1f}")


if __name__ == "__main__":
    main()
