#!/usr/bin/env python3
"""
Two plots for a given checkpoint:

Plot 1: For each token i in 0..N-2, compute e_i^T W_k^T W_q e_{i+1}
         where e is the token embedding and W_q, W_k are Layer 2's
         query/key weight matrices. This is the Layer 2 attention score
         of token i+1 (query) attending to token i (key).

Plot 2: Same but replacing embeddings with W_v1 @ e (Layer 1 value projection).
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
ARGS = parser.parse_args()


@torch.no_grad()
def main():
    print(f"Loading {ARGS.ckpt}")
    model = load_model_from_checkpoint(ARGS.ckpt)
    n_embd = model.config.n_embd
    vocab_n = model.config.vocab_size - 1
    block_size = model.config.block_size

    output_path = ARGS.output or os.path.join(
        os.path.dirname(ARGS.ckpt), "..", "plots", "qk_neighbor_score.png"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    layer1_attn = model.transformer.h[0].attn
    layer2_attn = model.transformer.h[1].attn

    W_q2 = layer2_attn.c_attn.weight[:n_embd, :]
    W_k2 = layer2_attn.c_attn.weight[n_embd:2*n_embd, :]
    W_v1 = layer1_attn.c_attn.weight[2*n_embd:, :]

    embeddings = model.transformer.wte.weight[:vocab_n, :]  # (vocab_n, n_embd)

    # Q = W_q2 @ e_i (query at sorted position showing value i)
    # K = W_k2 @ e_{i+1} (key for the next value)
    # score = Q^T K = e_i^T W_q2^T W_k2 e_{i+1}
    Q_raw = embeddings @ W_q2.T  # (vocab_n, n_embd)
    K_raw = embeddings @ W_k2.T  # (vocab_n, n_embd)

    # score(i) = Q_raw[i] . K_raw[i+1]  for i in 0..N-2
    scores_raw = (Q_raw[:-1] * K_raw[1:]).sum(dim=1).cpu().numpy()

    # Same with v = W_v1 @ e
    v_transformed = embeddings @ W_v1.T  # (vocab_n, n_embd)
    Q_v1 = v_transformed @ W_q2.T
    K_v1 = v_transformed @ W_k2.T
    scores_v1 = (Q_v1[:-1] * K_v1[1:]).sum(dim=1).cpu().numpy()

    token_values = np.arange(vocab_n - 1)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax = axes[0]
    ax.plot(token_values, scores_raw, linewidth=0.8, color="#2166ac")
    ax.set_ylabel(r"$e_i^\top W_Q^\top W_K\, e_{i+1}$", fontsize=12)
    ax.set_title("Layer 2 QK score: token i (query) vs token i+1 (key) — raw embeddings",
                 fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.2, linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[1]
    ax.plot(token_values, scores_v1, linewidth=0.8, color="#b2182b")
    ax.set_xlabel("Token value i", fontsize=11)
    ax.set_ylabel(r"$v_i^\top W_Q^\top W_K\, v_{i+1}$  where $v = W_{V_1} e$", fontsize=12)
    ax.set_title("Layer 2 QK score: token i (query) vs i+1 (key) — Layer 1 value-projected embeddings",
                 fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.2, linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle(
        f"k={block_size}, N={vocab_n}, checkpoint={os.path.basename(ARGS.ckpt)}",
        fontsize=11, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
