#!/usr/bin/env python3
"""
Check if monotonicity exists in the raw embedding space.

Plots:
1) score = e_i^T W_Q2^T W_K2 e_j  (pure embeddings, no LN)
2) score = (W_v1 e_i)^T W_Q2^T W_K2 e_j  (value-projected query, raw key)
3) Same as 1 but with LN on both sides
4) score using the c_proj path

For each: slope plot + full profile + local centered.
Also a diagnostic: PCA of embeddings to check if they're ordered.
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
        os.path.dirname(ARGS.ckpt), "..", "plots", "qk_raw_structure.png"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    block0 = model.transformer.h[0]
    block1 = model.transformer.h[1]

    W_q2 = block1.attn.c_attn.weight[:n_embd, :]
    W_k2 = block1.attn.c_attn.weight[n_embd:2*n_embd, :]
    b_q2 = block1.attn.c_attn.bias[:n_embd]
    b_k2 = block1.attn.c_attn.bias[n_embd:2*n_embd]

    W_v1 = block0.attn.c_attn.weight[2*n_embd:, :]
    b_v1 = block0.attn.c_attn.bias[2*n_embd:]

    e = model.transformer.wte.weight[:vocab_n]  # (vocab_n, n_embd)

    # ---------- PCA of embeddings ----------
    e_np = e.cpu().numpy()
    e_centered = e_np - e_np.mean(axis=0)
    U, S, Vt = np.linalg.svd(e_centered, full_matrices=False)
    pc2 = e_centered @ Vt[:2].T  # (vocab_n, 2)

    # ---------- Score matrices ----------
    # (1) Raw: e_i vs e_j, no LN
    Q1 = e @ W_q2.T + b_q2
    K1 = e @ W_k2.T + b_k2
    S1 = (Q1 @ K1.T).cpu().numpy()

    # (2) Q = W_v1 @ e_i, K = e_j, no LN
    v_proj = e @ W_v1.T + b_v1
    Q2 = v_proj @ W_q2.T + b_q2
    K2 = K1  # same keys
    S2 = (Q2 @ K2.T).cpu().numpy()

    # (3) With LN on both sides: Q = LN(e), K = LN(e)
    ln_e = block1.ln_1(e)
    Q3 = ln_e @ W_q2.T + b_q2
    K3 = ln_e @ W_k2.T + b_k2
    S3 = (Q3 @ K3.T).cpu().numpy()

    # (4) Q from residual path (e + c_proj(W_v @ LN_L1(e))), K from LN(e)
    ln_e0 = block0.ln_1(e)
    val = ln_e0 @ W_v1.T + b_v1
    val_proj = val @ block0.attn.c_proj.weight.T + block0.attn.c_proj.bias
    x_res = e + val_proj
    v_res = block1.ln_1(x_res)
    Q4 = v_res @ W_q2.T + b_q2
    K4 = K3
    S4 = (Q4 @ K4.T).cpu().numpy()

    labels = [
        "Raw e (no LN)",
        "Q=W_v1@e, K=e",
        "Both LN'd",
        "Q=residual, K=LN(e)",
    ]
    score_mats = [S1, S2, S3, S4]
    colors = ["#1b7837", "#2166ac", "#762a83", "#b2182b"]

    fig = plt.figure(figsize=(22, 18))

    # Row 0: PCA of embeddings (spans full width)
    ax_pca = fig.add_axes([0.06, 0.82, 0.88, 0.14])
    scatter = ax_pca.scatter(pc2[:, 0], pc2[:, 1], c=np.arange(vocab_n),
                             cmap="viridis", s=4, alpha=0.7)
    ax_pca.set_xlabel("PC1")
    ax_pca.set_ylabel("PC2")
    ax_pca.set_title(f"PCA of token embeddings (color = token value), N={vocab_n}", fontweight="bold")
    plt.colorbar(scatter, ax=ax_pca, label="Token value")

    # Rows 1-3: slope, full profile, centered local
    gs_rows = [
        fig.add_axes([0.06 + 0.23*c, 0.55, 0.20, 0.21]) for c in range(4)
    ]
    gs_mid = [
        fig.add_axes([0.06 + 0.23*c, 0.30, 0.20, 0.21]) for c in range(4)
    ]
    gs_bot = [
        fig.add_axes([0.06 + 0.23*c, 0.04, 0.20, 0.21]) for c in range(4)
    ]

    for col, (label, S, color) in enumerate(zip(labels, score_mats, colors)):
        # Slope
        ax = gs_rows[col]
        slopes = S[np.arange(1, vocab_n-1), np.arange(2, vocab_n)] - \
                 S[np.arange(1, vocab_n-1), np.arange(0, vocab_n-2)]
        ax.plot(np.arange(1, vocab_n-1), slopes, linewidth=0.5, color=color, alpha=0.8)
        ax.axhline(y=0, color="black", linewidth=0.5)
        n_pos = np.sum(slopes > 0)
        n_neg = np.sum(slopes < 0)
        ax.set_title(f"{label}\n+:{n_pos}  −:{n_neg}", fontsize=9, fontweight="bold")
        ax.set_xlabel("i")
        ax.set_ylabel("score(i,i+1)−score(i,i−1)")
        ax.grid(True, alpha=0.15)

        # Full profile
        ax = gs_mid[col]
        for qi in [vocab_n//6, vocab_n//3, vocab_n//2, 2*vocab_n//3, 5*vocab_n//6]:
            ax.plot(np.arange(vocab_n), S[qi, :], linewidth=0.7, alpha=0.6, label=f"i={qi}")
            ax.axvline(x=qi, linestyle=":", linewidth=0.3, alpha=0.3)
        ax.set_xlabel("j")
        ax.set_ylabel("Score")
        ax.legend(fontsize=6, ncol=2)
        ax.grid(True, alpha=0.15)

        # Centered local
        ax = gs_bot[col]
        W = 15
        all_centered = []
        for qi in range(W+2, vocab_n-W-2):
            local = np.array([S[qi, qi+off] for off in range(-W, W+1)])
            local = local - local[W]
            all_centered.append(local)
            if qi % 10 == 0:
                ax.plot(range(-W, W+1), local, linewidth=0.2, alpha=0.1, color=color)
        all_centered = np.array(all_centered)
        mu = np.mean(all_centered, axis=0)
        ax.plot(range(-W, W+1), mu, linewidth=2.5, color="black", label="Mean")
        ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.7)
        ax.axvline(x=1, color="red", linestyle="--", linewidth=0.5, alpha=0.5)
        ax.set_xlabel("j−i")
        ax.set_ylabel("Score−self")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.15)

    fig.suptitle(
        f"Raw QK structure diagnostic — k={block_size}, N={vocab_n}, "
        f"{os.path.basename(ARGS.ckpt)}",
        fontsize=13, fontweight="bold", y=0.99,
    )
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
