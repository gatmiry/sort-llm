#!/usr/bin/env python3
"""
Heatmap of QK score varying (t, y) — the key-side base token and
L1 attention target — for a fixed query (z, x).

  Q = W_Q_L2 @ LN_L2( MLP1( LN_2( e_z + V_x ) ) )       [fixed]
  K(t,y) = W_K_L2 @ LN_L2( MLP1( LN_2( e_t + V_y ) ) )  [varies]

  score(t, y) = Q · K(t, y)

Axes: x-axis = y (L1 target at key), y-axis = t (key base token).
Multiple subplots for different fixed (z, x) choices.
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
        os.path.dirname(ARGS.ckpt), "..", "plots", "qk_heatmap_ty.png"
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

    def compute_Q(z, x):
        """Fixed query vector for base z, L1 target x."""
        inp = e_all[z] + V_all[x]  # (C,)
        mlp_out = block0.mlp(block0.ln_2(inp.unsqueeze(0)))  # (1, C)
        h = block1.ln_1(mlp_out)
        return (h @ W_q.T + b_q).squeeze(0)  # (C,)

    def compute_K_matrix(vocab_n):
        """Precompute K(t, y) for all (t, y).
        Returns shape (N, N, C) but that's too large.
        Instead, return a function that computes score for fixed Q."""
        pass

    def compute_heatmap_ty(Q_vec):
        """For a fixed Q, compute score(t, y) = Q · K(t,y) for all t, y.
        K(t,y) = W_K @ LN(MLP(LN_2(e_t + V_y)))
        We compute row-by-row (for each t)."""
        hm = np.zeros((vocab_n, vocab_n))
        batch_size = 64
        for t_start in range(0, vocab_n, batch_size):
            t_end = min(t_start + batch_size, vocab_n)
            for t in range(t_start, t_end):
                # e_t + V_y for all y: (N, C)
                inp = e_all[t].unsqueeze(0) + V_all  # (N, C)
                mlp_out = block0.mlp(block0.ln_2(inp))  # (N, C)
                h = block1.ln_1(mlp_out)
                K = h @ W_k.T + b_k  # (N, C)
                scores = (K @ Q_vec).cpu().numpy()  # (N,)
                hm[t, :] = scores
        return hm

    # Fixed (z, x) choices — representative cases
    # z = query base (token at query position), x = L1 attention target at query
    configs = [
        (250, 250, "z=250, x=250\n(self-attend)"),
        (250, 251, "z=250, x=251\n(L1→next)"),
        (250, 249, "z=250, x=249\n(L1→prev)"),
        (100, 100, "z=100, x=100\n(self-attend)"),
        (100, 101, "z=100, x=101\n(L1→next)"),
        (400, 401, "z=400, x=401\n(L1→next)"),
    ]

    n_plots = len(configs)
    n_cols = 3
    n_rows = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7.5 * n_cols, 6.5 * n_rows))
    axes = axes.flatten()

    for idx, (z, x, label) in enumerate(configs):
        print(f"  Computing heatmap for {label.split(chr(10))[0]} ...")
        Q_vec = compute_Q(z, x)
        hm = compute_heatmap_ty(Q_vec)

        ax = axes[idx]
        vmax = np.percentile(np.abs(hm), 99)
        im = ax.imshow(hm, aspect="auto", origin="lower",
                       cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                       interpolation="nearest")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, format="%.0f")

        ax.set_xlabel("y (key-side L1 target)", fontsize=10)
        ax.set_ylabel("t (key base token)", fontsize=10)
        ax.set_title(label, fontsize=10, fontweight="bold")

        # Mark the diagonal t = y
        ax.plot([0, vocab_n-1], [0, vocab_n-1], "w--", linewidth=0.5, alpha=0.5)
        # Mark z and x
        ax.axvline(x, color="lime", linewidth=0.6, linestyle="--", alpha=0.5, label=f"x={x}")
        ax.axhline(z, color="cyan", linewidth=0.6, linestyle="--", alpha=0.5, label=f"z={z}")

        # Analysis: which axis matters more?
        var_over_y = np.var(hm, axis=1).mean()  # avg variance across y for fixed t
        var_over_t = np.var(hm, axis=0).mean()  # avg variance across t for fixed y
        total_var = np.var(hm)

        # Argmax for t=z (the "natural" key base)
        row_z = hm[z, :]
        best_y = np.argmax(row_z)

        # For each t, where is argmax y?
        argmax_y = np.argmax(hm, axis=1)
        # How often is argmax_y close to t+1?
        n_next = np.sum(argmax_y[:-1] == np.arange(vocab_n-1) + 1)

        ann = (f"Var across y|t: {var_over_y:.0f}\n"
               f"Var across t|y: {var_over_t:.0f}\n"
               f"best_y(t={z}): {best_y}\n"
               f"argmax_y=t+1: {n_next}/{vocab_n-1}")
        ax.text(0.02, 0.98, ann, transform=ax.transAxes, fontsize=7,
                verticalalignment="top", fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))

    fig.suptitle(
        f"QK score heatmap: score(t, y) for fixed query (z, x)\n"
        f"Q = W_Q·LN(MLP(LN(e_z+V_x))),  K(t,y) = W_K·LN(MLP(LN(e_t+V_y)))\n"
        f"y-axis = t (key token), x-axis = y (key L1 target)\n"
        f"k={block_size}, N={vocab_n}, {os.path.basename(ARGS.ckpt)}",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")

    # Print summary
    print("\n=== Summary ===")
    for idx, (z, x, label) in enumerate(configs):
        Q_vec = compute_Q(z, x)
        hm = compute_heatmap_ty(Q_vec)
        var_y = np.var(hm, axis=1).mean()
        var_t = np.var(hm, axis=0).mean()
        print(f"  {label.split(chr(10))[0]:20s}: "
              f"Var(y|t)={var_y:.0f}, Var(t|y)={var_t:.0f}, "
              f"ratio t/y = {var_t/var_y:.2f}")


if __name__ == "__main__":
    main()
