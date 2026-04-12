#!/usr/bin/env python3
"""
Like qk_cross_mean, but include positional embeddings:

Query side: v_i = processing through Layer 1 of (e_i + pos_{block_size + i_sorted_pos})
  where i_sorted_pos = position in the output region of the sorted sequence.
  Sorted output starts at position block_size in the 2*block_size-length sequence.

Key side: w_j = LN_block1( e_j + pos_p )
  where p ranges over positions in the input region [0..block_size).

Simulates what the actual model sees: query is at a sorted-output position,
keys are at unsorted-input positions.
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
parser.add_argument("--window", type=int, default=25)
parser.add_argument("--key-pos", type=int, default=0,
                    help="Which input-region position to use for key (0..block_size-1). "
                         "Use -1 to average over all input positions.")
ARGS = parser.parse_args()


@torch.no_grad()
def main():
    print(f"Loading {ARGS.ckpt}")
    model = load_model_from_checkpoint(ARGS.ckpt)
    n_embd = model.config.n_embd
    vocab_n = model.config.vocab_size - 1
    block_size = model.config.block_size

    output_path = ARGS.output or os.path.join(
        os.path.dirname(ARGS.ckpt), "..", "plots", "qk_cross_with_pos.png"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    block0 = model.transformer.h[0]
    block1 = model.transformer.h[1]

    W_q2 = block1.attn.c_attn.weight[:n_embd, :]
    W_k2 = block1.attn.c_attn.weight[n_embd:2*n_embd, :]

    wte = model.transformer.wte.weight[:vocab_n]  # (vocab_n, n_embd)
    wpe = model.transformer.wpe.weight             # (2*block_size, n_embd)

    W = ARGS.window
    margin = W + 2

    # Query positions: output region starts at block_size.
    # We'll use the first few sorted positions.
    # The query at sorted position p wants value rank p.
    # For simplicity, for query token i, we put it at position (block_size + sorted_pos)
    # where sorted_pos ∈ [1..block_size-1] (pos 0 is the separator or first sorted output).
    # We'll try a few query positions to see the effect.

    query_positions = [block_size + p for p in [1, block_size//4, block_size//2, 3*block_size//4]]
    qp_labels = [f"sorted pos {p-block_size}" for p in query_positions]

    # Key positions: input region [0..block_size)
    if ARGS.key_pos == -1:
        key_pos_list = list(range(block_size))
    else:
        key_pos_list = [ARGS.key_pos]

    # Precompute K for each key position, then average
    K_avg = torch.zeros(vocab_n, n_embd, device=DEVICE)
    for kp in key_pos_list:
        w_j = block1.ln_1(wte + wpe[kp])
        K_avg += w_j @ W_k2.T
    K_avg /= len(key_pos_list)

    offsets = np.arange(-W, W + 1)

    fig, axes = plt.subplots(2, len(query_positions), figsize=(4.5 * len(query_positions), 9),
                             squeeze=False)

    for col, (qpos, qlab) in enumerate(zip(query_positions, qp_labels)):
        pos_emb_q = wpe[qpos]

        # v_i without MLP
        e_q = (wte + pos_emb_q).unsqueeze(1)  # (vocab_n, 1, n_embd)
        h = block0.ln_1(e_q)
        attn_out = block0.attn(h)
        x_no_mlp = e_q + attn_out
        v_no_mlp = block1.ln_1(x_no_mlp).squeeze(1)
        Q_no_mlp = v_no_mlp @ W_q2.T

        # v_i with MLP
        x_with_mlp = x_no_mlp.clone()
        x_with_mlp = x_with_mlp + block0.mlp(block0.ln_2(x_with_mlp))
        v_with_mlp = block1.ln_1(x_with_mlp).squeeze(1)
        Q_with_mlp = v_with_mlp @ W_q2.T

        score_no_mlp = (Q_no_mlp @ K_avg.T).cpu().numpy()
        score_with_mlp = (Q_with_mlp @ K_avg.T).cpu().numpy()

        valid_tokens = list(range(margin, vocab_n - margin))

        for row, (score_mat, label, color) in enumerate([
            (score_no_mlp, "v without MLP", "#2166ac"),
            (score_with_mlp, "v with MLP", "#b2182b"),
        ]):
            ax = axes[row][col]

            curves = []
            for qi in valid_tokens:
                curve = [score_mat[qi, qi + off] for off in offsets]
                curves.append(curve)
                ax.plot(offsets, curve, alpha=0.05, linewidth=0.4, color=color)

            curves = np.array(curves)
            mu = np.mean(curves, axis=0)
            std = np.std(curves, axis=0)
            ax.plot(offsets, mu, color="black", linewidth=2.5, label="Mean")
            ax.fill_between(offsets, mu - std, mu + std, color="gray", alpha=0.15)

            ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.7, alpha=0.6)
            ax.axvline(x=1, color="red", linestyle="--", linewidth=0.5, alpha=0.5)

            if row == 0:
                ax.set_title(f"{qlab}", fontsize=10, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"Score ({label})")
            ax.set_xlabel("j − i")
            ax.legend(fontsize=8, loc="upper right")
            ax.grid(True, alpha=0.15)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    kp_label = "avg over input positions" if ARGS.key_pos == -1 else f"key_pos={ARGS.key_pos}"
    fig.suptitle(
        f"$(W_Q \\, v_i)^\\top (W_K \\, w_j)$ with positional embeddings\n"
        f"$w_j = LN(e_j + pos_{{key}})$, $v_i = e_i + pos_{{query}}$ processed thru L1\n"
        f"k={block_size}, N={vocab_n}, {os.path.basename(ARGS.ckpt)} | {kp_label}",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.89])
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
