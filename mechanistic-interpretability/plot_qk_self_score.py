#!/usr/bin/env python3
"""
Two plots for a given checkpoint:

Plot 1: For each token v in the vocabulary, compute e_v^T W_k^T W_q e_v
         where e_v is the token embedding and W_q, W_k are Layer 2's
         query/key weight matrices. This is the Layer 2 self-attention
         score of each token with itself.

Plot 2: Same but replacing e_v with W_v1 @ e_v, where W_v1 is Layer 1's
         value weight matrix (the component added to the residual stream
         by Layer 1 attention).
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
parser.add_argument("--output", default=None, help="Output path (default: plots dir next to ckpt)")
ARGS = parser.parse_args()


@torch.no_grad()
def main():
    print(f"Loading {ARGS.ckpt}")
    model = load_model_from_checkpoint(ARGS.ckpt)
    n_embd = model.config.n_embd
    vocab_n = model.config.vocab_size - 1  # exclude SEP
    block_size = model.config.block_size

    output_path = ARGS.output or os.path.join(
        os.path.dirname(ARGS.ckpt), "..", "plots", "qk_self_score.png"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Extract weight matrices
    # c_attn.weight shape: (3*n_embd, n_embd) — maps input -> [Q, K, V]
    layer1_attn = model.transformer.h[0].attn
    layer2_attn = model.transformer.h[1].attn

    W_q2 = layer2_attn.c_attn.weight[:n_embd, :]        # (n_embd, n_embd)
    W_k2 = layer2_attn.c_attn.weight[n_embd:2*n_embd, :]  # (n_embd, n_embd)
    W_v1 = layer1_attn.c_attn.weight[2*n_embd:, :]       # (n_embd, n_embd)

    # M = W_k2^T @ W_q2, so score = e^T M e
    M2 = W_k2.T @ W_q2  # (n_embd, n_embd)

    # Token embeddings for vocab tokens 0..vocab_n-1
    embeddings = model.transformer.wte.weight[:vocab_n, :]  # (vocab_n, n_embd)

    # Plot 1: e_v^T M2 e_v for raw embeddings
    # (vocab_n, n_embd) @ (n_embd, n_embd) -> (vocab_n, n_embd)
    Me = (embeddings @ M2.T)  # each row i = e_i^T M2^T, but since score is scalar: e^T M e = (M e) . e
    scores_raw = (Me * embeddings).sum(dim=1).cpu().numpy()

    # Plot 2: v = W_v1 @ e, then v^T M2 v
    # v_i = W_v1 @ e_i -> (n_embd,)
    v_transformed = embeddings @ W_v1.T  # (vocab_n, n_embd): each row = W_v1 @ e_i
    Mv = (v_transformed @ M2.T)
    scores_v1 = (Mv * v_transformed).sum(dim=1).cpu().numpy()

    token_values = np.arange(vocab_n)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax = axes[0]
    ax.plot(token_values, scores_raw, linewidth=0.8, color="#2166ac")
    ax.set_ylabel(r"$e_v^\top\, W_K^\top W_Q\, e_v$", fontsize=12)
    ax.set_title("Layer 2 QK self-score on raw token embeddings", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.2, linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[1]
    ax.plot(token_values, scores_v1, linewidth=0.8, color="#b2182b")
    ax.set_xlabel("Token value", fontsize=11)
    ax.set_ylabel(r"$v^\top\, W_K^\top W_Q\, v$  where $v = W_{V_1}\, e$", fontsize=12)
    ax.set_title(
        "Layer 2 QK self-score on Layer 1 value-projected embeddings",
        fontsize=13, fontweight="bold",
    )
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
