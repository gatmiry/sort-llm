#!/usr/bin/env python3
"""
Heatmap of QK score varying (x, t) — the query-side L1 target and
key base token — for fixed (z, y).

  Q(x) = W_Q_L2 @ LN_L2( MLP1( LN_2( e_z + V_x ) ) )   [x varies, z fixed]
  K(t) = W_K_L2 @ LN_L2( MLP1( LN_2( e_t + V_y ) ) )   [t varies, y fixed]

  score(x, t) = Q(x) · K(t)

Axes: x-axis = t (key base token), y-axis = x (query L1 target).
Multiple subplots for different fixed (z, y) choices.
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
        os.path.dirname(ARGS.ckpt), "..", "plots", "qk_heatmap_xt.png"
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

    def compute_all_Q_vary_x(z):
        """Q(x) for all x, with fixed query base z. Returns (N, C)."""
        inp = e_all[z].unsqueeze(0) + V_all  # (N, C) — e_z + V_x for all x
        mlp_out = block0.mlp(block0.ln_2(inp))
        h = block1.ln_1(mlp_out)
        return h @ W_q.T + b_q  # (N, C)

    def compute_all_K_vary_t(y):
        """K(t) for all t, with fixed key L1 target y. Returns (N, C)."""
        inp = e_all + V_all[y].unsqueeze(0)  # (N, C) — e_t + V_y for all t
        mlp_out = block0.mlp(block0.ln_2(inp))
        h = block1.ln_1(mlp_out)
        return h @ W_k.T + b_k  # (N, C)

    def compute_heatmap_xt(z, y):
        """score(x, t) = Q(x; z) · K(t; y) for all x, t."""
        Q = compute_all_Q_vary_x(z)  # (N, C)
        K = compute_all_K_vary_t(y)  # (N, C)
        return (Q @ K.T).cpu().numpy()  # (N, N): rows=x, cols=t

    configs = [
        (250, 250, "z=250, y=250\n(same token)"),
        (250, 251, "z=250, y=251\n(y = z+1)"),
        (251, 250, "z=251, y=250\n(y = z−1)"),
        (100, 100, "z=100, y=100\n(same token)"),
        (100, 101, "z=100, y=101\n(y = z+1)"),
        (400, 401, "z=400, y=401\n(y = z+1)"),
    ]

    n_cols = 3
    n_rows = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7.5 * n_cols, 6.5 * n_rows))
    axes = axes.flatten()

    for idx, (z, y, label) in enumerate(configs):
        print(f"  Computing heatmap for {label.split(chr(10))[0]} ...")
        hm = compute_heatmap_xt(z, y)

        ax = axes[idx]
        vmax = np.percentile(np.abs(hm), 99)
        if vmax < 1:
            vmax = 1
        im = ax.imshow(hm, aspect="auto", origin="lower",
                       cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                       interpolation="nearest")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, format="%.0f")

        ax.set_xlabel("t (key base token)", fontsize=10)
        ax.set_ylabel("x (query L1 target)", fontsize=10)
        ax.set_title(label, fontsize=10, fontweight="bold")

        ax.plot([0, vocab_n-1], [0, vocab_n-1], "w--", linewidth=0.5, alpha=0.5)
        ax.axhline(y, color="cyan", linewidth=0.6, linestyle="--", alpha=0.6)
        ax.axvline(z, color="lime", linewidth=0.6, linestyle="--", alpha=0.6)

        var_over_t = np.var(hm, axis=1).mean()  # variance across t for fixed x
        var_over_x = np.var(hm, axis=0).mean()  # variance across x for fixed t

        argmax_t = np.argmax(hm, axis=1)  # for each x, best t
        n_next = np.sum(argmax_t[:-1] == np.arange(vocab_n - 1) + 1)

        ann = (f"Var across t|x: {var_over_t:.0f}\n"
               f"Var across x|t: {var_over_x:.0f}\n"
               f"ratio t/x: {var_over_t / max(var_over_x, 1):.1f}\n"
               f"argmax_t=x+1: {n_next}/{vocab_n-1}")
        ax.text(0.02, 0.98, ann, transform=ax.transAxes, fontsize=7,
                verticalalignment="top", fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))

    fig.suptitle(
        f"QK score heatmap: score(x, t) for fixed (z, y)\n"
        f"Q(x) = W_Q·LN(MLP(LN(e_z+V_x))),  K(t) = W_K·LN(MLP(LN(e_t+V_y)))\n"
        f"y-axis = x (query L1 target), x-axis = t (key base token)\n"
        f"k={block_size}, N={vocab_n}, {os.path.basename(ARGS.ckpt)}",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
