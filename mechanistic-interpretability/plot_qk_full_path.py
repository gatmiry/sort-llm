#!/usr/bin/env python3
"""
Compute v for each token by tracing the value path through Layer 1:

  e  →  LN_1 of block 0  →  W_v of block 0 attn  →  LN_2 of block 0
     →  MLP of block 0  →  LN_1 of block 1  →  v

Then plot:
  Plot 1 (self):     v_i^T  W_Q^T W_K  v_i
  Plot 2 (neighbor):  v_i^T  W_Q^T W_K  v_{i+1}

where W_Q, W_K are Layer 2 query/key weights.
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
        os.path.dirname(ARGS.ckpt), "..", "plots", "qk_full_value_path.png"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    block0 = model.transformer.h[0]
    block1 = model.transformer.h[1]

    # Layer 2 Q/K weights
    W_q2 = block1.attn.c_attn.weight[:n_embd, :]
    W_k2 = block1.attn.c_attn.weight[n_embd:2*n_embd, :]

    # Layer 1 value weights + bias
    W_v1 = block0.attn.c_attn.weight[2*n_embd:, :]
    b_v1 = block0.attn.c_attn.bias[2*n_embd:]

    # Raw embeddings (vocab_n, n_embd)
    e = model.transformer.wte.weight[:vocab_n, :]

    # Step 1: LN before first attention
    h = block0.ln_1(e)

    # Step 2: multiply by value weight of Layer 1 attention
    val = h @ W_v1.T + b_v1  # (vocab_n, n_embd)

    # Step 3: LN before MLP of block 0
    h2 = block0.ln_2(val)

    # Step 4: MLP of block 0
    h3 = block0.mlp(h2)

    # Step 5: LN before attention of block 1
    v = block1.ln_1(h3)

    # Compute Q and K projections
    Q = v @ W_q2.T  # (vocab_n, n_embd)
    K = v @ W_k2.T  # (vocab_n, n_embd)

    # Self-score: v_i^T W_Q^T W_K v_i
    scores_self = (Q * K).sum(dim=1).cpu().numpy()

    # Neighbor score: v_i^T W_Q^T W_K v_{i+1} (query=i, key=i+1)
    scores_neighbor = (Q[:-1] * K[1:]).sum(dim=1).cpu().numpy()

    token_values = np.arange(vocab_n)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax = axes[0]
    ax.plot(token_values, scores_self, linewidth=0.8, color="#2166ac")
    ax.set_ylabel("Self-score", fontsize=11)
    ax.set_title(
        r"$v_i^\top\, W_Q^\top W_K\, v_i$  (self-score)",
        fontsize=13, fontweight="bold",
    )
    ax.grid(True, alpha=0.2, linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[1]
    ax.plot(token_values[:-1], scores_neighbor, linewidth=0.8, color="#b2182b")
    ax.set_xlabel("Token value i", fontsize=11)
    ax.set_ylabel("Neighbor score", fontsize=11)
    ax.set_title(
        r"$v_i^\top\, W_Q^\top W_K\, v_{i+1}$  (query=i, key=i+1)",
        fontsize=13, fontweight="bold",
    )
    ax.grid(True, alpha=0.2, linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle(
        f"Layer 2 QK scores with v = LN₁∘MLP₁∘LN₂∘W_V₁∘LN₁(embedding)\n"
        f"k={block_size}, N={vocab_n}, {os.path.basename(ARGS.ckpt)}",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
