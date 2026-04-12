#!/usr/bin/env python3
"""
Heatmap of score(i,j) for the 4 v_i versions.
Zoomed both to full vocabulary and to a local region.
If monotonic, heatmap should show asymmetry around the diagonal.
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
        os.path.dirname(ARGS.ckpt), "..", "plots", "qk_heatmap.png"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    block0 = model.transformer.h[0]
    block1 = model.transformer.h[1]
    attn1 = block1.attn

    W_q2 = attn1.c_attn.weight[:n_embd, :]
    W_k2 = attn1.c_attn.weight[n_embd:2*n_embd, :]
    b_q2 = attn1.c_attn.bias[:n_embd]
    b_k2 = attn1.c_attn.bias[n_embd:2*n_embd]

    W_v1 = block0.attn.c_attn.weight[2*n_embd:, :]
    b_v1 = block0.attn.c_attn.bias[2*n_embd:]

    e = model.transformer.wte.weight[:vocab_n]

    # Key side: w_j = LN_block1(e_j)
    w = block1.ln_1(e)
    K = w @ W_k2.T + b_k2

    # v_i version A: just W_v (no c_proj, no MLP)
    ln0_e = block0.ln_1(e)
    val_raw = ln0_e @ W_v1.T + b_v1
    v_A = block1.ln_1(e + val_raw)

    # v_i version B: with c_proj (no MLP)
    val_proj = val_raw @ block0.attn.c_proj.weight.T + block0.attn.c_proj.bias
    v_B = block1.ln_1(e + val_proj)

    # v_i version C: full block0 with MLP
    e_3d = e.unsqueeze(1)
    h = block0.ln_1(e_3d)
    attn_out = block0.attn(h)
    x_C = e_3d + attn_out
    if block0.mlp is not None:
        x_C = x_C + block0.mlp(block0.ln_2(x_C))
    v_C = block1.ln_1(x_C).squeeze(1)

    versions = [
        ("W_v only", v_A),
        ("+ c_proj", v_B),
        ("+ MLP (full block0)", v_C),
    ]

    fig, axes = plt.subplots(len(versions), 3, figsize=(18, 5 * len(versions)))

    center = vocab_n // 2
    zoom = 40  # half-window for zoomed view

    for row, (label, v_i) in enumerate(versions):
        Q = v_i @ W_q2.T + b_q2
        S = (Q @ K.T).cpu().numpy()

        # Col 0: full heatmap
        ax = axes[row][0]
        vmax = np.percentile(np.abs(S), 98)
        im = ax.imshow(S, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                       origin="lower", aspect="auto")
        ax.plot([0, vocab_n-1], [0, vocab_n-1], color="black", linewidth=0.5, alpha=0.5)
        ax.set_xlabel("j (key)")
        ax.set_ylabel("i (query)")
        ax.set_title(f"{label} — full", fontweight="bold", fontsize=10)
        plt.colorbar(im, ax=ax, shrink=0.7)

        # Col 1: zoomed around center
        ax = axes[row][1]
        S_zoom = S[center-zoom:center+zoom, center-zoom:center+zoom]
        im = ax.imshow(S_zoom, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                       origin="lower", aspect="auto",
                       extent=[center-zoom, center+zoom, center-zoom, center+zoom])
        ax.plot([center-zoom, center+zoom], [center-zoom, center+zoom],
                color="black", linewidth=0.5, alpha=0.5)
        ax.set_xlabel("j")
        ax.set_ylabel("i")
        ax.set_title(f"Zoomed [{center-zoom}:{center+zoom}]", fontweight="bold", fontsize=10)
        plt.colorbar(im, ax=ax, shrink=0.7)

        # Col 2: anti-diagonal score: S[i, i+k] - S[i, i-k]
        # If monotonic, this should be consistently positive (or negative)
        ax = axes[row][2]
        for k_val in [1, 2, 3, 5, 10]:
            asymm = []
            valid_range = range(k_val, vocab_n - k_val)
            for i in valid_range:
                asymm.append(S[i, i + k_val] - S[i, i - k_val])
            asymm = np.array(asymm)
            frac_pos = np.mean(asymm > 0) * 100
            ax.plot(list(valid_range), asymm, linewidth=0.5, alpha=0.6,
                    label=f"k={k_val} ({frac_pos:.0f}%+)")
        ax.axhline(y=0, color="black", linewidth=0.5)
        ax.set_xlabel("i")
        ax.set_ylabel("S[i,i+k] − S[i,i−k]")
        ax.set_title(f"Asymmetry: S(i,i+k)−S(i,i−k)", fontweight="bold", fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.15)

    fig.suptitle(
        f"QK score heatmaps — k={block_size}, N={vocab_n}, "
        f"{os.path.basename(ARGS.ckpt)}\n"
        f"Q side: $v_i$ (processed), K side: $w_j = LN(e_j)$",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
