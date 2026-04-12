#!/usr/bin/env python3
"""
Focused plot: how fraction of consistent-direction asymmetry varies
with offset k and with/without MLP. Confirms monotonic structure.
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
def compute_versions(model):
    n_embd = model.config.n_embd
    vocab_n = model.config.vocab_size - 1

    block0 = model.transformer.h[0]
    block1 = model.transformer.h[1]

    W_q2 = block1.attn.c_attn.weight[:n_embd, :]
    W_k2 = block1.attn.c_attn.weight[n_embd:2*n_embd, :]
    b_q2 = block1.attn.c_attn.bias[:n_embd]
    b_k2 = block1.attn.c_attn.bias[n_embd:2*n_embd]

    W_v1 = block0.attn.c_attn.weight[2*n_embd:, :]
    b_v1 = block0.attn.c_attn.bias[2*n_embd:]

    e = model.transformer.wte.weight[:vocab_n]

    # Key side always: w_j = LN_block1(e_j)
    w = block1.ln_1(e)
    K = w @ W_k2.T + b_k2

    # V1: just LN(e) on query side (no L1 at all)
    v0 = block1.ln_1(e)
    Q0 = v0 @ W_q2.T + b_q2
    S0 = (Q0 @ K.T).cpu().numpy()

    # V2: W_v only (no c_proj)
    ln0_e = block0.ln_1(e)
    val_raw = ln0_e @ W_v1.T + b_v1
    v_A = block1.ln_1(e + val_raw)
    QA = v_A @ W_q2.T + b_q2
    SA = (QA @ K.T).cpu().numpy()

    # V3: with c_proj
    val_proj = val_raw @ block0.attn.c_proj.weight.T + block0.attn.c_proj.bias
    v_B = block1.ln_1(e + val_proj)
    QB = v_B @ W_q2.T + b_q2
    SB = (QB @ K.T).cpu().numpy()

    # V4: full block0 with MLP
    e_3d = e.unsqueeze(1)
    h = block0.ln_1(e_3d)
    attn_out = block0.attn(h)
    x_C = e_3d + attn_out
    if block0.mlp is not None:
        x_C = x_C + block0.mlp(block0.ln_2(x_C))
    v_C = block1.ln_1(x_C).squeeze(1)
    QC = v_C @ W_q2.T + b_q2
    SC = (QC @ K.T).cpu().numpy()

    return [
        ("LN(e) only", S0),
        ("+ W_v (no c_proj)", SA),
        ("+ W_v + c_proj", SB),
        ("+ full block0 (with MLP)", SC),
    ], vocab_n


def asymmetry_curve(S, vocab_n, max_k=30):
    """For each k, compute fraction of i where S[i,i+k] > S[i,i-k]."""
    ks = list(range(1, max_k + 1))
    fracs = []
    for k in ks:
        pos = 0
        total = 0
        for i in range(k, vocab_n - k):
            if S[i, i + k] > S[i, i - k]:
                pos += 1
            total += 1
        fracs.append(pos / total * 100)
    return ks, fracs


@torch.no_grad()
def main():
    model = load_model_from_checkpoint(ARGS.ckpt)
    vocab_n = model.config.vocab_size - 1
    block_size = model.config.block_size

    output_path = ARGS.output or os.path.join(
        os.path.dirname(ARGS.ckpt), "..", "plots", "qk_monotonicity_summary.png"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    versions, vocab_n = compute_versions(model)

    colors = ["#999999", "#1b7837", "#2166ac", "#b2182b"]
    max_k = min(40, vocab_n // 4)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: fraction positive asymmetry vs k
    ax = axes[0]
    for (label, S), color in zip(versions, colors):
        ks, fracs = asymmetry_curve(S, vocab_n, max_k)
        ax.plot(ks, fracs, linewidth=2, color=color, label=label, marker=".", markersize=4)
    ax.axhline(y=50, color="black", linestyle="--", linewidth=0.5, alpha=0.5, label="Random (50%)")
    ax.set_xlabel("Offset k", fontsize=11)
    ax.set_ylabel("% of i where S[i,i+k] > S[i,i−k]", fontsize=11)
    ax.set_title("Monotonicity: fraction with consistent direction", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.15)
    ax.set_ylim(30, 100)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Right: mean asymmetry S[i,i+k] - S[i,i-k] vs k (normalized)
    ax = axes[1]
    for (label, S), color in zip(versions, colors):
        ks = list(range(1, max_k + 1))
        mean_asymm = []
        for k in ks:
            vals = [S[i, i + k] - S[i, i - k] for i in range(k, vocab_n - k)]
            mean_asymm.append(np.mean(vals))
        ax.plot(ks, mean_asymm, linewidth=2, color=color, label=label, marker=".", markersize=4)
    ax.axhline(y=0, color="black", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.set_xlabel("Offset k", fontsize=11)
    ax.set_ylabel("Mean S[i,i+k] − S[i,i−k]", fontsize=11)
    ax.set_title("Mean asymmetry magnitude", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle(
        f"Monotonic QK structure — k={block_size}, N={vocab_n}, "
        f"{os.path.basename(ARGS.ckpt)}\n"
        f"Query: $v_i$ (different processing), Key: $w_j = LN(e_j)$",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
