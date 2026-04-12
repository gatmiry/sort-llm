#!/usr/bin/env python3
"""
Score: (W_Q v_i)^T (W_K w_j)  where:
  w_j = LN₁_block1(e_j)             — raw embedding through LN before Layer 2
  v_i = processed through Layer 1    — two versions:
    (a) no MLP:  LN₁_L2( e + Attn_L1(LN₁_L1(e)) )
    (b) with MLP: LN₁_L2( block0(e) )              — full block 0

Plot as function of j zoomed around i, for several fixed i values.
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
parser.add_argument("--query-tokens", type=int, nargs="+", default=None)
parser.add_argument("--window", type=int, default=25, help="Half-window around query token")
ARGS = parser.parse_args()


@torch.no_grad()
def main():
    print(f"Loading {ARGS.ckpt}")
    model = load_model_from_checkpoint(ARGS.ckpt)
    n_embd = model.config.n_embd
    vocab_n = model.config.vocab_size - 1
    block_size = model.config.block_size

    output_path = ARGS.output or os.path.join(
        os.path.dirname(ARGS.ckpt), "..", "plots", "qk_cross_local.png"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    block0 = model.transformer.h[0]
    block1 = model.transformer.h[1]

    W_q2 = block1.attn.c_attn.weight[:n_embd, :]
    W_k2 = block1.attn.c_attn.weight[n_embd:2*n_embd, :]

    e = model.transformer.wte.weight[:vocab_n]  # (vocab_n, n_embd)

    # --- w_j: raw embedding through LN before Layer 2 ---
    w = block1.ln_1(e)  # (vocab_n, n_embd)
    K_w = w @ W_k2.T    # (vocab_n, n_embd)

    # --- v_i version (a): no MLP ---
    e_3d = e.unsqueeze(1)  # (vocab_n, 1, n_embd)
    h = block0.ln_1(e_3d)
    attn_out = block0.attn(h)
    x_no_mlp = e_3d + attn_out
    v_no_mlp = block1.ln_1(x_no_mlp).squeeze(1)  # (vocab_n, n_embd)
    Q_no_mlp = v_no_mlp @ W_q2.T

    # --- v_i version (b): with MLP ---
    x_with_mlp = x_no_mlp.clone()
    if block0.mlp is not None:
        x_with_mlp = x_with_mlp + block0.mlp(block0.ln_2(x_with_mlp))
    v_with_mlp = block1.ln_1(x_with_mlp).squeeze(1)
    Q_with_mlp = v_with_mlp @ W_q2.T

    if ARGS.query_tokens:
        query_tokens = ARGS.query_tokens
    else:
        margin = ARGS.window + 5
        step = (vocab_n - 2 * margin) // 5
        query_tokens = [margin + step * k for k in range(6)]

    W = ARGS.window
    n_queries = len(query_tokens)

    fig, axes = plt.subplots(2, n_queries, figsize=(4.2 * n_queries, 7), sharex=False)

    for col, qi in enumerate(query_tokens):
        j_lo = max(0, qi - W)
        j_hi = min(vocab_n, qi + W + 1)
        j_range = np.arange(j_lo, j_hi)
        offsets = j_range - qi

        # (a) no MLP
        scores_a = (Q_no_mlp[qi].unsqueeze(0) * K_w[j_lo:j_hi]).sum(dim=1).cpu().numpy()
        # (b) with MLP
        scores_b = (Q_with_mlp[qi].unsqueeze(0) * K_w[j_lo:j_hi]).sum(dim=1).cpu().numpy()

        for row, (scores, label, color) in enumerate([
            (scores_a, "v without MLP", "#2166ac"),
            (scores_b, "v with MLP", "#b2182b"),
        ]):
            ax = axes[row][col]
            ax.bar(offsets, scores, width=0.8, color=color, alpha=0.7)
            ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.7, alpha=0.6)
            ax.axvline(x=1, color="red", linestyle="--", linewidth=0.7, alpha=0.6)

            if row == 0:
                ax.set_title(f"query i={qi}", fontsize=10, fontweight="bold")
            if row == 1:
                ax.set_xlabel("j − i", fontsize=9)
            if col == 0:
                ax.set_ylabel(f"Score ({label})", fontsize=9)

            ax.grid(True, axis="y", alpha=0.15, linestyle=":")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(labelsize=7)

    fig.suptitle(
        f"$(W_Q v_i)^\\top (W_K w_j)$   where  $w_j = LN(e_j)$,  "
        f"$v_i$ = Layer 1 processed\n"
        f"k={block_size}, N={vocab_n}, {os.path.basename(ARGS.ckpt)}  "
        f"(gray=i, red=i+1)",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
