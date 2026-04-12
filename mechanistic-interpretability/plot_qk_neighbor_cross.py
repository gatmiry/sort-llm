#!/usr/bin/env python3
"""
Neighbor score (i vs i+1) and self-score (i vs i) using asymmetric definitions:
  Query side v_i: processed through Layer 1 (two versions: ± MLP)
  Key side w_j:   raw embedding through LN before Layer 2

Plots score(i) for ALL i, not fixing i.
"""

import os, sys, argparse
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
    model = load_model_from_checkpoint(ARGS.ckpt)
    n_embd = model.config.n_embd
    vocab_n = model.config.vocab_size - 1
    block_size = model.config.block_size

    output_path = ARGS.output or os.path.join(
        os.path.dirname(ARGS.ckpt), "..", "plots", "qk_neighbor_cross.png"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    block0 = model.transformer.h[0]
    block1 = model.transformer.h[1]

    W_q2 = block1.attn.c_attn.weight[:n_embd, :]
    W_k2 = block1.attn.c_attn.weight[n_embd:2*n_embd, :]
    b_q2 = block1.attn.c_attn.bias[:n_embd]
    b_k2 = block1.attn.c_attn.bias[n_embd:2*n_embd]

    e = model.transformer.wte.weight[:vocab_n]

    # --- Key side: w_j = LN_block1(e_j) for all j ---
    w = block1.ln_1(e)
    K = w @ W_k2.T + b_k2  # (vocab_n, n_embd)

    # --- Query side: v_i without MLP ---
    # e + attn_block0(LN_block0(e))  where attn = c_proj(W_v @ h + b_v)
    e_3d = e.unsqueeze(1)
    h = block0.ln_1(e_3d)
    attn_out = block0.attn(h)       # c_proj(W_v @ LN(e) + b_v)
    x_no_mlp = (e_3d + attn_out).squeeze(1)
    v_no = block1.ln_1(x_no_mlp)    # LN before L2 attention
    Q_no = v_no @ W_q2.T + b_q2

    # --- Query side: v_i with MLP ---
    x_mlp = e_3d + attn_out
    x_mlp = x_mlp + block0.mlp(block0.ln_2(x_mlp))
    v_yes = block1.ln_1(x_mlp.squeeze(1))
    Q_yes = v_yes @ W_q2.T + b_q2

    # --- Also build K from processed v (same as Q processing) ---
    # Without MLP: key side also uses v
    K_vno = v_no @ W_k2.T + b_k2
    # With MLP: key side also uses v
    K_vyes = v_yes @ W_k2.T + b_k2

    token_i = np.arange(vocab_n)
    token_i_n = np.arange(vocab_n - 1)

    # Compute all scores:
    # (A) v_i vs w_{i+1}  (processed Q, raw K)
    # (B) v_i vs v_{i+1}  (both processed)
    results = {}
    for mlp_label, Q, K_raw, K_v in [
        ("without MLP", Q_no, K, K_vno),
        ("with MLP", Q_yes, K, K_vyes),
    ]:
        self_vw  = (Q * K_raw).sum(dim=1).cpu().numpy()
        neigh_vw = (Q[:-1] * K_raw[1:]).sum(dim=1).cpu().numpy()
        self_vv  = (Q * K_v).sum(dim=1).cpu().numpy()
        neigh_vv = (Q[:-1] * K_v[1:]).sum(dim=1).cpu().numpy()
        results[mlp_label] = {
            "self_vw": self_vw, "neigh_vw": neigh_vw,
            "self_vv": self_vv, "neigh_vv": neigh_vv,
        }

    fig, axes = plt.subplots(3, 4, figsize=(24, 13), sharex=True)
    col_cfgs = [
        ("without MLP", "v×w", "self_vw", "neigh_vw", "#2166ac"),
        ("without MLP", "v×v", "self_vv", "neigh_vv", "#4393c3"),
        ("with MLP", "v×w", "self_vw", "neigh_vw", "#b2182b"),
        ("with MLP", "v×v", "self_vv", "neigh_vv", "#d6604d"),
    ]

    for col, (mlp_label, pair_label, sk, nk, color) in enumerate(col_cfgs):
        r = results[mlp_label]
        s_scores = r[sk]
        n_scores = r[nk]
        diff = n_scores - s_scores[:-1]

        axes[0][col].plot(token_i, s_scores, linewidth=0.7, color=color)
        axes[0][col].set_title(f"Self — {mlp_label}, {pair_label}", fontsize=10, fontweight="bold")
        axes[0][col].set_ylabel("Self-score", fontsize=9)
        axes[0][col].grid(True, alpha=0.15, linestyle=":")
        axes[0][col].spines["top"].set_visible(False)
        axes[0][col].spines["right"].set_visible(False)

        axes[1][col].plot(token_i_n, n_scores, linewidth=0.7, color=color)
        axes[1][col].set_title(f"Neighbor (i→i+1) — {mlp_label}, {pair_label}", fontsize=10, fontweight="bold")
        axes[1][col].set_ylabel("Neighbor score", fontsize=9)
        axes[1][col].grid(True, alpha=0.15, linestyle=":")
        axes[1][col].spines["top"].set_visible(False)
        axes[1][col].spines["right"].set_visible(False)

        n_pos = np.sum(diff > 0)
        n_neg = np.sum(diff < 0)
        axes[2][col].plot(token_i_n, diff, linewidth=0.7, color=color)
        axes[2][col].axhline(y=0, color="black", linewidth=0.5, alpha=0.5)
        axes[2][col].set_title(f"Neigh−Self — {mlp_label}, {pair_label}  (+:{n_pos} −:{n_neg})",
                               fontsize=10, fontweight="bold")
        axes[2][col].set_ylabel("Neighbor − Self", fontsize=9)
        axes[2][col].set_xlabel("Token i", fontsize=9)
        axes[2][col].grid(True, alpha=0.15, linestyle=":")
        axes[2][col].spines["top"].set_visible(False)
        axes[2][col].spines["right"].set_visible(False)

    fig.suptitle(
        f"$v_i$ vs $w_{{i+1}}$ (raw key)  and  $v_i$ vs $v_{{i+1}}$ (processed key)\n"
        f"k={block_size}, N={vocab_n}, {os.path.basename(ARGS.ckpt)}",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
