#!/usr/bin/env python3
"""
Same as local instances but with smoothed curves to reveal monotonic trend.
Both raw (thin) and smoothed (bold) overlaid.
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
parser.add_argument("--window", type=int, default=40)
parser.add_argument("--smooth", type=int, default=5, help="Smoothing half-window")
parser.add_argument("--n-instances", type=int, default=12)
ARGS = parser.parse_args()


def smooth(y, hw):
    out = np.convolve(y, np.ones(2*hw+1)/(2*hw+1), mode="same")
    return out


@torch.no_grad()
def main():
    model = load_model_from_checkpoint(ARGS.ckpt)
    n_embd = model.config.n_embd
    vocab_n = model.config.vocab_size - 1
    block_size = model.config.block_size

    output_path = ARGS.output or os.path.join(
        os.path.dirname(ARGS.ckpt), "..", "plots", "qk_local_smooth.png"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    block0 = model.transformer.h[0]
    block1 = model.transformer.h[1]

    W_q2 = block1.attn.c_attn.weight[:n_embd, :]
    W_k2 = block1.attn.c_attn.weight[n_embd:2*n_embd, :]
    b_q2 = block1.attn.c_attn.bias[:n_embd]
    b_k2 = block1.attn.c_attn.bias[n_embd:2*n_embd]

    e = model.transformer.wte.weight[:vocab_n]

    w = block1.ln_1(e)
    K = w @ W_k2.T + b_k2

    e_3d = e.unsqueeze(1)
    h = block0.ln_1(e_3d)
    attn_out = block0.attn(h)

    # Without MLP
    x_no = (e_3d + attn_out).squeeze(1)
    v_no = block1.ln_1(x_no)
    Q_no = v_no @ W_q2.T + b_q2
    S_no = (Q_no @ K.T).cpu().numpy()

    # With MLP
    x_yes = e_3d + attn_out
    x_yes = x_yes + block0.mlp(block0.ln_2(x_yes))
    v_yes = block1.ln_1(x_yes.squeeze(1))
    Q_yes = v_yes @ W_q2.T + b_q2
    S_yes = (Q_yes @ K.T).cpu().numpy()

    W = ARGS.window
    margin = W + ARGS.smooth + 2
    n = ARGS.n_instances
    step = (vocab_n - 2 * margin) // (n - 1)
    query_tokens = [margin + step * k for k in range(n)]

    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 4 * nrows))
    axes = np.atleast_2d(axes)

    offsets = np.arange(-W, W + 1)
    hw = ARGS.smooth

    for idx, qi in enumerate(query_tokens):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]

        raw_no = np.array([S_no[qi, qi + off] for off in offsets])
        raw_yes = np.array([S_yes[qi, qi + off] for off in offsets])

        sm_no = smooth(raw_no, hw)
        sm_yes = smooth(raw_yes, hw)

        ax.plot(offsets, raw_no, linewidth=0.4, color="#2166ac", alpha=0.35)
        ax.plot(offsets, raw_yes, linewidth=0.4, color="#b2182b", alpha=0.35)
        ax.plot(offsets, sm_no, linewidth=2.2, color="#2166ac", label="without MLP")
        ax.plot(offsets, sm_yes, linewidth=2.2, color="#b2182b", label="with MLP")

        ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.6, alpha=0.4)
        ax.set_title(f"i = {qi}", fontsize=10, fontweight="bold")
        ax.set_xlabel("j − i", fontsize=9)
        if c == 0:
            ax.set_ylabel("Score", fontsize=9)
        ax.grid(True, alpha=0.12, linestyle=":")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=8)
        if idx == 0:
            ax.legend(fontsize=7, loc="best")

    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].set_visible(False)

    fig.suptitle(
        f"$(W_Q\\,v_i)^\\top(W_K\\,w_j)$  local around $i$  "
        f"(smoothed, half-window={hw})\n"
        f"$w_j = LN(e_j)$,  $v_i$: L1 attn value path ± MLP\n"
        f"k={block_size}, N={vocab_n}, {os.path.basename(ARGS.ckpt)}",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
