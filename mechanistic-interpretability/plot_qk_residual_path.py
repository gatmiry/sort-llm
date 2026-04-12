#!/usr/bin/env python3
"""
Define v by running each token embedding through the actual Layer 1 attention
(with residual), skipping MLP, then through LN before Layer 2:

  v = LN₁_block1( e + Attn_block0( LN₁_block0(e) ) )

For a single token, self-attention weight = 1, so
  Attn(x) = c_proj(W_v x + b_v)

Then plot self-score and neighbor-score using Layer 2 Q/K weights.
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
        os.path.dirname(ARGS.ckpt), "..", "plots", "qk_residual_path.png"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    block0 = model.transformer.h[0]
    block1 = model.transformer.h[1]

    W_q2 = block1.attn.c_attn.weight[:n_embd, :]
    W_k2 = block1.attn.c_attn.weight[n_embd:2*n_embd, :]

    # Batch of single-token sequences: (vocab_n, 1, n_embd)
    e = model.transformer.wte.weight[:vocab_n].unsqueeze(1)

    # LN before first attention
    h = block0.ln_1(e)

    # Layer 1 attention (single token attends to itself with weight 1)
    attn_out = block0.attn(h)

    # Residual connection: embedding + attention output
    x = e + attn_out

    # LN before Layer 2 attention
    v = block1.ln_1(x).squeeze(1)  # (vocab_n, n_embd)

    # Layer 2 Q/K projections
    Q = v @ W_q2.T
    K = v @ W_k2.T

    scores_self = (Q * K).sum(dim=1).cpu().numpy()
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
        f"v = LN₁_L2( e + Attn_L1( LN₁_L1(e) ) )   [no MLP]\n"
        f"k={block_size}, N={vocab_n}, {os.path.basename(ARGS.ckpt)}",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
