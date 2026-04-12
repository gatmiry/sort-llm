#!/usr/bin/env python3
"""
Simple plot: score(i, j) vs j around i, for several fixed i values.
Two v_i versions (with/without MLP), w_j = LN(e_j).
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
parser.add_argument("--window", type=int, default=20)
parser.add_argument("--n-instances", type=int, default=12)
ARGS = parser.parse_args()


@torch.no_grad()
def main():
    model = load_model_from_checkpoint(ARGS.ckpt)
    n_embd = model.config.n_embd
    vocab_n = model.config.vocab_size - 1
    block_size = model.config.block_size

    output_path = ARGS.output or os.path.join(
        os.path.dirname(ARGS.ckpt), "..", "plots", "qk_local_instances.png"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    block0 = model.transformer.h[0]
    block1 = model.transformer.h[1]

    W_q2 = block1.attn.c_attn.weight[:n_embd, :]
    W_k2 = block1.attn.c_attn.weight[n_embd:2*n_embd, :]
    b_q2 = block1.attn.c_attn.bias[:n_embd]
    b_k2 = block1.attn.c_attn.bias[n_embd:2*n_embd]

    e = model.transformer.wte.weight[:vocab_n]

    # Keys: w_j = LN_block1(e_j)
    w = block1.ln_1(e)
    K = w @ W_k2.T + b_k2  # (vocab_n, n_embd)

    # v_i WITHOUT MLP: e + attn_L1(single token) → LN_block1
    e_3d = e.unsqueeze(1)
    h = block0.ln_1(e_3d)
    attn_out = block0.attn(h)
    x_no_mlp = (e_3d + attn_out).squeeze(1)
    v_no_mlp = block1.ln_1(x_no_mlp)
    Q_no_mlp = v_no_mlp @ W_q2.T + b_q2
    S_no_mlp = (Q_no_mlp @ K.T).cpu().numpy()

    # v_i WITH MLP: full block0 → LN_block1
    x_mlp = e_3d + attn_out
    x_mlp = x_mlp + block0.mlp(block0.ln_2(x_mlp))
    v_mlp = block1.ln_1(x_mlp.squeeze(1))
    Q_mlp = v_mlp @ W_q2.T + b_q2
    S_mlp = (Q_mlp @ K.T).cpu().numpy()

    W = ARGS.window
    margin = W + 5
    n = ARGS.n_instances
    step = (vocab_n - 2 * margin) // (n - 1)
    query_tokens = [margin + step * k for k in range(n)]

    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.8 * nrows))
    axes = np.atleast_2d(axes)

    offsets = np.arange(-W, W + 1)

    for idx, qi in enumerate(query_tokens):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]

        s_no = [S_no_mlp[qi, qi + off] for off in offsets]
        s_yes = [S_mlp[qi, qi + off] for off in offsets]

        ax.plot(offsets, s_no, linewidth=1.5, color="#2166ac", label="without MLP", marker=".", markersize=3)
        ax.plot(offsets, s_yes, linewidth=1.5, color="#b2182b", label="with MLP", marker=".", markersize=3)
        ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.6, alpha=0.5)
        ax.set_title(f"i = {qi}", fontsize=10, fontweight="bold")
        ax.set_xlabel("j − i", fontsize=9)
        if c == 0:
            ax.set_ylabel("Score", fontsize=9)
        ax.grid(True, alpha=0.15, linestyle=":")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=8)
        if idx == 0:
            ax.legend(fontsize=8)

    # Hide unused subplots
    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].set_visible(False)

    fig.suptitle(
        f"$(W_Q\\,v_i)^\\top(W_K\\,w_j)$  local around $i$\n"
        f"$w_j = LN(e_j)$,  $v_i$: L1 attn value path ± MLP\n"
        f"k={block_size}, N={vocab_n}, {os.path.basename(ARGS.ckpt)}",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
