#!/usr/bin/env python3
"""
Extended QK heatmap with base embeddings (z, t).

For fixed tokens z (query base) and t (key base):
  Q_x = W_Q_L2 @ LN_L2( MLP1( LN_2( e_z + V_x ) ) )
  K_y = W_K_L2 @ LN_L2( MLP1( LN_2( e_t + V_y ) ) )
  score(x, y | z, t) = Q_x · K_y

V_x = c_proj_L1(W_v_L1 @ LN_1(e_x) + b_v)  (L1 value vector for token x)

Draw heatmaps for multiple (z, t) pairs.
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
        os.path.dirname(ARGS.ckpt), "..", "plots", "qk_value_heatmap_base.png"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Precompute L1 value vectors for all tokens
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

    def compute_heatmap(z, t):
        """Compute score(x, y | z, t) for all x, y."""
        e_z = e_all[z].unsqueeze(0).expand(vocab_n, -1)  # (N, C)
        e_t = e_all[t].unsqueeze(0).expand(vocab_n, -1)

        # Query: MLP(LN_2(e_z + V_x)) for all x
        mlp_q = block0.mlp(block0.ln_2(e_z + V_all))
        h_q = block1.ln_1(mlp_q)
        Q = h_q @ W_q.T + b_q  # (N, C)

        # Key: MLP(LN_2(e_t + V_y)) for all y
        mlp_k = block0.mlp(block0.ln_2(e_t + V_all))
        h_k = block1.ln_1(mlp_k)
        K = h_k @ W_k.T + b_k  # (N, C)

        return (Q @ K.T).cpu().numpy()

    # Grid of (z, t) pairs
    # Row 0: z=250, t varies relative to z
    # Row 1: z=100, t varies relative to z
    # Row 2: z=400, t varies relative to z
    # Columns: t - z = -2, -1, 0, +1, +2, +5
    z_values = [100, 250, 400]
    offsets = [-2, -1, 0, 1, 2, 5]

    n_rows = len(z_values)
    n_cols = len(offsets)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 4.2 * n_rows))

    for row, z in enumerate(z_values):
        for col, off in enumerate(offsets):
            t = z + off
            t = max(0, min(t, vocab_n - 1))
            ax = axes[row][col]

            hm = compute_heatmap(z, t)

            # Zoom into region around z and t
            center = (z + t) // 2
            win = 40
            lo = max(0, center - win)
            hi = min(vocab_n, center + win)
            hm_zoom = hm[lo:hi, lo:hi]

            vmax = np.percentile(np.abs(hm_zoom), 98)
            if vmax < 1:
                vmax = 1
            im = ax.imshow(hm_zoom, aspect="auto", origin="lower",
                         cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                         extent=[lo, hi, lo, hi], interpolation="nearest")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, format="%.0f")

            # Mark z and t
            ax.axhline(z, color="lime", linewidth=0.8, linestyle="--", alpha=0.7)
            ax.axvline(t, color="cyan", linewidth=0.8, linestyle="--", alpha=0.7)
            # Mark diagonal
            diag_lo = max(lo, lo)
            diag_hi = min(hi, hi)
            ax.plot([diag_lo, diag_hi], [diag_lo, diag_hi], "k--", linewidth=0.4, alpha=0.3)

            # Argmax analysis in zoomed region
            argmax_offsets = []
            for xi in range(lo, hi):
                best_y = lo + np.argmax(hm[xi, lo:hi])
                argmax_offsets.append(best_y - xi)
            med_off = np.median(argmax_offsets)

            ax.set_title(f"z={z}, t={t} (t−z={off})\nmed argmax off={med_off:.0f}",
                         fontsize=9, fontweight="bold")
            ax.set_xlabel("Key: y (L1→y)", fontsize=8)
            ax.set_ylabel("Query: x (L1→x)", fontsize=8)
            ax.tick_params(labelsize=7)

    fig.suptitle(
        f"QK heatmap with base embeddings: score(x,y | z,t)\n"
        f"Q = W_Q·LN(MLP(LN(e_z+V_x))),  K = W_K·LN(MLP(LN(e_t+V_y)))\n"
        f"Green line = z (query base), Cyan line = t (key base)\n"
        f"k={block_size}, N={vocab_n}, {os.path.basename(ARGS.ckpt)}",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")

    # ---- Additional analysis: extract key statistics per (z,t) ----
    print("\n=== Detailed analysis ===")
    for z in z_values:
        for off in offsets:
            t = max(0, min(z + off, vocab_n - 1))
            hm = compute_heatmap(z, t)

            self_score = np.mean([hm[i, i] for i in range(vocab_n)])
            next_score = np.mean([hm[i, i+1] for i in range(vocab_n - 1)])
            prev_score = np.mean([hm[i, i-1] for i in range(1, vocab_n)])

            # Score at (z, t) specifically
            score_zt = hm[z, t]
            # Score at (z, z)
            score_zz = hm[z, z]
            # Score at (t, t)
            score_tt = hm[t, t] if t < vocab_n else 0

            # Argmax for query=z
            best_y_for_z = np.argmax(hm[z, :])
            # Argmax for key=t
            best_x_for_t = np.argmax(hm[:, t])

            # How many x have argmax(y) = x+1?
            n_next = sum(1 for i in range(vocab_n-1) if np.argmax(hm[i, :]) == i+1)

            print(f"  z={z}, t={t} (off={off:+d}): "
                  f"s(z,t)={score_zt:.0f}, s(z,z)={score_zz:.0f}, "
                  f"best_y(z)={best_y_for_z} (off={best_y_for_z-z:+d}), "
                  f"best_x(t)={best_x_for_t} (off={best_x_for_t-t:+d}), "
                  f"n_argmax=x+1: {n_next}/{vocab_n-1}")


if __name__ == "__main__":
    main()
