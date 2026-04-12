#!/usr/bin/env python3
"""
Fix a query token i and plot its QK score against all other tokens j.

For each of the 3 definitions of v:
  1. Raw embedding:  v = e
  2. W_v only:       v = W_v1 @ LN₁_L1(e)
  3. Residual path:  v = LN₁_L2( e + Attn_L1( LN₁_L1(e) ) )

Plot v_i^T W_Q^T W_K v_j  as a function of j, for several choices of i.
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
parser.add_argument("--query-tokens", type=int, nargs="+", default=None,
                    help="Token values to use as fixed query (default: spread across vocab)")
ARGS = parser.parse_args()


@torch.no_grad()
def main():
    print(f"Loading {ARGS.ckpt}")
    model = load_model_from_checkpoint(ARGS.ckpt)
    n_embd = model.config.n_embd
    vocab_n = model.config.vocab_size - 1
    block_size = model.config.block_size

    output_path = ARGS.output or os.path.join(
        os.path.dirname(ARGS.ckpt), "..", "plots", "qk_fixed_query.png"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    block0 = model.transformer.h[0]
    block1 = model.transformer.h[1]

    W_q2 = block1.attn.c_attn.weight[:n_embd, :]
    W_k2 = block1.attn.c_attn.weight[n_embd:2*n_embd, :]
    W_v1 = block0.attn.c_attn.weight[2*n_embd:, :]
    b_v1 = block0.attn.c_attn.bias[2*n_embd:]

    e = model.transformer.wte.weight[:vocab_n]  # (vocab_n, n_embd)

    # --- Definition 1: raw embedding ---
    Q1 = e @ W_q2.T
    K1 = e @ W_k2.T

    # --- Definition 2: W_v1 @ LN(e) ---
    h = block0.ln_1(e)
    v2 = h @ W_v1.T + b_v1
    Q2 = v2 @ W_q2.T
    K2 = v2 @ W_k2.T

    # --- Definition 3: residual path (no MLP) ---
    e3 = e.unsqueeze(1)  # (vocab_n, 1, n_embd)
    h3 = block0.ln_1(e3)
    attn_out = block0.attn(h3)
    x3 = e3 + attn_out
    v3 = block1.ln_1(x3).squeeze(1)  # (vocab_n, n_embd)
    Q3 = v3 @ W_q2.T
    K3 = v3 @ W_k2.T

    if ARGS.query_tokens:
        query_tokens = ARGS.query_tokens
    else:
        step = vocab_n // 6
        query_tokens = [step * k for k in range(6)]

    definitions = [
        ("Raw embedding", Q1, K1),
        (r"$W_{V_1} \cdot LN(e)$", Q2, K2),
        ("Residual: e + Attn₁(LN(e)), then LN", Q3, K3),
    ]

    n_defs = len(definitions)
    n_queries = len(query_tokens)
    fig, axes = plt.subplots(n_defs, n_queries, figsize=(4.5 * n_queries, 3.5 * n_defs),
                             sharex=True)

    colors = plt.cm.Set1(np.linspace(0, 0.8, n_queries))
    j_values = np.arange(vocab_n)

    for row, (def_name, Q, K) in enumerate(definitions):
        for col, qi in enumerate(query_tokens):
            ax = axes[row][col]
            # score(j) = Q[qi] . K[j]
            scores = (Q[qi].unsqueeze(0) * K).sum(dim=1).cpu().numpy()

            ax.plot(j_values, scores, linewidth=0.6, color=colors[col], alpha=0.8)
            ax.axvline(x=qi, color="gray", linestyle="--", linewidth=0.7, alpha=0.5)
            ax.axvline(x=qi + 1, color="red", linestyle="--", linewidth=0.7, alpha=0.5)

            if row == 0:
                ax.set_title(f"query i={qi}", fontsize=10, fontweight="bold")
            if row == n_defs - 1:
                ax.set_xlabel("key token j", fontsize=9)
            if col == 0:
                ax.set_ylabel(def_name, fontsize=9)

            ax.grid(True, alpha=0.15, linestyle=":")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(labelsize=7)

    fig.suptitle(
        f"$v_i^\\top W_Q^\\top W_K\\, v_j$ for fixed query i, varying key j\n"
        f"k={block_size}, N={vocab_n}, {os.path.basename(ARGS.ckpt)}  "
        f"(gray dashed=i, red dashed=i+1)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
