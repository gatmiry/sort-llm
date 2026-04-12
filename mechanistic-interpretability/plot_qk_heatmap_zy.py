#!/usr/bin/env python3
"""
Heatmap of QK score varying (z, y) — the query base token and
key-side L1 attention target — for fixed (x, t).

  Q(z) = W_Q_L2 @ LN_L2( MLP1( LN_2( e_z + V_x ) ) )   [z varies, x fixed]
  K(y) = W_K_L2 @ LN_L2( MLP1( LN_2( e_t + V_y ) ) )   [y varies, t fixed]

  score(z, y) = Q(z) · K(y)

Axes: x-axis = y (key L1 target), y-axis = z (query base token).
Multiple subplots for different fixed (x, t) choices.
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

    block0 = model.transformer.h[0]
    block1 = model.transformer.h[1]

    output_path = ARGS.output or os.path.join(
        os.path.dirname(ARGS.ckpt), "..", "plots", "qk_heatmap_zy.png"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    e_all = model.transformer.wte.weight[:vocab_n]  # (N, C)
    ln1_e = block0.ln_1(e_all)
    W_v = block0.attn.c_attn.weight[2*n_embd:3*n_embd, :]
    b_v = block0.attn.c_attn.bias[2*n_embd:3*n_embd]
    v = ln1_e @ W_v.T + b_v
    V_all = v @ block0.attn.c_proj.weight.T + block0.attn.c_proj.bias  # (N, C)

    W_q = block1.attn.c_attn.weight[:n_embd, :]
    b_q = block1.attn.c_attn.bias[:n_embd]
    W_k = block1.attn.c_attn.weight[n_embd:2*n_embd, :]
    b_k = block1.attn.c_attn.bias[n_embd:2*n_embd]

    def compute_all_Q(x):
        """Q(z) for all z, with fixed L1 target x. Returns (N, C)."""
        inp = e_all + V_all[x].unsqueeze(0)  # (N, C) — e_z + V_x for all z
        mlp_out = block0.mlp(block0.ln_2(inp))
        h = block1.ln_1(mlp_out)
        return h @ W_q.T + b_q  # (N, C)

    def compute_all_K(t):
        """K(y) for all y, with fixed key base t. Returns (N, C)."""
        inp = e_all[t].unsqueeze(0) + V_all  # (N, C) — e_t + V_y for all y
        mlp_out = block0.mlp(block0.ln_2(inp))
        h = block1.ln_1(mlp_out)
        return h @ W_k.T + b_k  # (N, C)

    def compute_heatmap_zy(x, t):
        """score(z, y) = Q(z; x) · K(y; t) for all z, y."""
        Q = compute_all_Q(x)  # (N, C)
        K = compute_all_K(t)  # (N, C)
        return (Q @ K.T).cpu().numpy()  # (N, N)

    # Fixed (x, t) choices
    configs = [
        (250, 250, "x=250, t=250\n(same token)"),
        (250, 251, "x=250, t=251\n(t = x+1)"),
        (251, 250, "x=251, t=250\n(t = x−1)"),
        (100, 100, "x=100, t=100\n(same token)"),
        (100, 101, "x=100, t=101\n(t = x+1)"),
        (400, 401, "x=400, t=401\n(t = x+1)"),
    ]

    n_cols = 3
    n_rows = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7.5 * n_cols, 6.5 * n_rows))
    axes = axes.flatten()

    for idx, (x, t, label) in enumerate(configs):
        print(f"  Computing heatmap for {label.split(chr(10))[0]} ...")
        hm = compute_heatmap_zy(x, t)

        ax = axes[idx]
        vmax = np.percentile(np.abs(hm), 99)
        if vmax < 1:
            vmax = 1
        im = ax.imshow(hm, aspect="auto", origin="lower",
                       cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                       interpolation="nearest")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, format="%.0f")

        ax.set_xlabel("y (key-side L1 target)", fontsize=10)
        ax.set_ylabel("z (query base token)", fontsize=10)
        ax.set_title(label, fontsize=10, fontweight="bold")

        # Mark diagonal z = y
        ax.plot([0, vocab_n-1], [0, vocab_n-1], "w--", linewidth=0.5, alpha=0.5)
        # Mark x and t
        ax.axvline(t, color="cyan", linewidth=0.6, linestyle="--", alpha=0.6)
        ax.axhline(x, color="lime", linewidth=0.6, linestyle="--", alpha=0.6)

        var_over_y = np.var(hm, axis=1).mean()
        var_over_z = np.var(hm, axis=0).mean()

        # For each z, where is argmax y?
        argmax_y = np.argmax(hm, axis=1)
        n_next = np.sum(argmax_y[:-1] == np.arange(vocab_n - 1) + 1)
        # How often is argmax_y = t?
        n_eq_t = np.sum(argmax_y == t)

        ann = (f"Var across y|z: {var_over_y:.0f}\n"
               f"Var across z|y: {var_over_z:.0f}\n"
               f"ratio z/y: {var_over_z / max(var_over_y, 1):.1f}\n"
               f"argmax_y=z+1: {n_next}/{vocab_n-1}\n"
               f"argmax_y=t: {n_eq_t}/{vocab_n}")
        ax.text(0.02, 0.98, ann, transform=ax.transAxes, fontsize=7,
                verticalalignment="top", fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))

    fig.suptitle(
        f"QK score heatmap: score(z, y) for fixed (x, t)\n"
        f"Q(z) = W_Q·LN(MLP(LN(e_z+V_x))),  K(y) = W_K·LN(MLP(LN(e_t+V_y)))\n"
        f"y-axis = z (query base token), x-axis = y (key L1 target)\n"
        f"k={block_size}, N={vocab_n}, {os.path.basename(ARGS.ckpt)}",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
