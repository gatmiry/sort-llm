#!/usr/bin/env python3
"""
Diagnostic plot for monotonicity of (W_Q v_i)^T (W_K w_j) around j=i.

1) Compute slope = score(i, i+1) - score(i, i-1) for ALL i. Plot it.
   If consistent sign → monotonicity is real and same direction.

2) Plot full-vocabulary profiles score(i, j) for a few selected i values,
   to see if the function is monotonic in j globally.

3) Also try the simpler v_i = just W_v @ LN(e_i) (no c_proj),
   to match "attn value weight" literally.
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
        os.path.dirname(ARGS.ckpt), "..", "plots", "qk_slope_profiles.png"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    block0 = model.transformer.h[0]
    block1 = model.transformer.h[1]

    attn0 = block0.attn
    attn1 = block1.attn

    W_q2 = attn1.c_attn.weight[:n_embd, :]
    W_k2 = attn1.c_attn.weight[n_embd:2*n_embd, :]
    b_q2 = attn1.c_attn.bias[:n_embd]
    b_k2 = attn1.c_attn.bias[n_embd:2*n_embd]

    W_v1 = attn0.c_attn.weight[2*n_embd:, :]
    b_v1 = attn0.c_attn.bias[2*n_embd:]
    W_proj1 = attn0.c_proj.weight
    b_proj1 = attn0.c_proj.bias

    e = model.transformer.wte.weight[:vocab_n]  # (vocab_n, n_embd)

    # --- w_j: raw embedding through LN before Layer 2 ---
    w = block1.ln_1(e)
    K_all = w @ W_k2.T + b_k2  # (vocab_n, n_embd) -- with bias

    # --- Build 4 versions of v_i ---
    # (A) "Just W_v": e + W_v1 @ LN_1(e)  -- no c_proj, no MLP
    ln1_e = block0.ln_1(e)
    val_raw = ln1_e @ W_v1.T + b_v1  # W_v applied to LN(e)
    # No c_proj — just the value vector added to residual
    x_A = e + val_raw
    v_A = block1.ln_1(x_A)

    # (B) "With c_proj": e + c_proj(W_v @ LN_1(e))  -- no MLP
    val_proj = val_raw @ W_proj1.T + b_proj1
    x_B = e + val_proj
    v_B = block1.ln_1(x_B)

    # (C) Full block0.attn on single tokens -- should match (B)
    e_3d = e.unsqueeze(1)
    h = block0.ln_1(e_3d)
    attn_out = block0.attn(h)  # includes c_proj
    x_C = e_3d + attn_out
    v_C = block1.ln_1(x_C).squeeze(1)

    # (D) Full block0 including MLP
    x_D = x_C.clone()
    if block0.mlp is not None:
        x_D = x_D + block0.mlp(block0.ln_2(x_D))
    v_D = block1.ln_1(x_D).squeeze(1)

    versions = [
        ("W_v only (no c_proj)", v_A, "#1b7837"),
        ("with c_proj (no MLP)", v_B, "#2166ac"),
        ("full attn (verify=B)", v_C, "#4393c3"),
        ("full block0 (+ MLP)", v_D, "#b2182b"),
    ]

    fig, axes = plt.subplots(3, len(versions), figsize=(5 * len(versions), 14))

    for col, (label, v_i, color) in enumerate(versions):
        Q_all = v_i @ W_q2.T + b_q2  # (vocab_n, n_embd)
        scores = (Q_all @ K_all.T).cpu().numpy()  # (vocab_n, vocab_n)

        # Row 0: Slope = score(i, i+1) - score(i, i-1)
        ax = axes[0][col]
        slopes = scores[np.arange(1, vocab_n - 1), np.arange(2, vocab_n)] - \
                 scores[np.arange(1, vocab_n - 1), np.arange(0, vocab_n - 2)]
        ax.plot(np.arange(1, vocab_n - 1), slopes, linewidth=0.6, color=color, alpha=0.8)
        ax.axhline(y=0, color="black", linewidth=0.5)
        n_pos = np.sum(slopes > 0)
        n_neg = np.sum(slopes < 0)
        ax.set_title(f"{label}\nslope>0: {n_pos}, slope<0: {n_neg}", fontsize=9, fontweight="bold")
        ax.set_xlabel("token i")
        ax.set_ylabel("score(i,i+1) − score(i,i−1)")
        ax.grid(True, alpha=0.15)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Row 1: Full profile score(i, j) for selected i, ALL j
        ax = axes[1][col]
        for qi in [vocab_n//6, vocab_n//3, vocab_n//2, 2*vocab_n//3, 5*vocab_n//6]:
            ax.plot(np.arange(vocab_n), scores[qi, :], linewidth=0.8, alpha=0.7,
                    label=f"i={qi}")
            ax.axvline(x=qi, linestyle=":", linewidth=0.4, alpha=0.4)
        ax.set_title(f"Full profile score(i,j) vs j", fontsize=9)
        ax.set_xlabel("j (token value)")
        ax.set_ylabel("Score")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.15)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Row 2: Zoomed local profile, overlaid and SORTED by direction
        ax = axes[2][col]
        W = 15
        for qi in range(W + 2, vocab_n - W - 2, max(1, vocab_n // 60)):
            local = [scores[qi, qi + off] for off in range(-W, W + 1)]
            local = np.array(local) - local[W]  # center at self-score
            ax.plot(np.arange(-W, W + 1), local, linewidth=0.3, alpha=0.15, color=color)

        # Mean centered profile
        all_centered = []
        for qi in range(W + 2, vocab_n - W - 2):
            local = [scores[qi, qi + off] for off in range(-W, W + 1)]
            local = np.array(local) - local[W]
            all_centered.append(local)
        all_centered = np.array(all_centered)
        mu = np.mean(all_centered, axis=0)
        ax.plot(np.arange(-W, W + 1), mu, linewidth=2.5, color="black", label="Mean")
        ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.7)
        ax.axvline(x=1, color="red", linestyle="--", linewidth=0.5, alpha=0.5)
        ax.set_title(f"Centered local (−self_score)", fontsize=9)
        ax.set_xlabel("j − i")
        ax.set_ylabel("Score − self_score")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.15)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        f"QK score monotonicity diagnostic — k={block_size}, N={vocab_n}\n"
        f"{os.path.basename(ARGS.ckpt)}  |  "
        f"$w_j = LN(e_j)$, different $v_i$ versions",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
