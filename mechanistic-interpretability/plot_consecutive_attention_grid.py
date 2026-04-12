#!/usr/bin/env python3
"""
Same as plot_consecutive_attention.py but draws each sample in its own subplot
in a grid layout, so individual attention patterns are clearly visible.
"""

import os
import sys
import random
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "sort-llm", "sortgpt_toolkit"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from model import DEVICE, load_model_from_checkpoint
from plot_heatmaps import get_all_layer_probs

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", required=True)
parser.add_argument("--output", default=None)
parser.add_argument("--n-samples", type=int, default=25)
parser.add_argument("--consec-len", type=int, default=6)
parser.add_argument("--threshold", type=float, default=0.05)
ARGS = parser.parse_args()

CONSEC_LEN = ARGS.consec_len
THRESHOLD = ARGS.threshold
N_SAMPLES = ARGS.n_samples


def generate_consecutive_batch(block_size, vocab_n, consec_len, device):
    while True:
        i_start = random.randint(0, vocab_n - consec_len)
        consec = list(range(i_start, i_start + consec_len))
        remaining_pool = [v for v in range(vocab_n) if v not in consec]
        if len(remaining_pool) < block_size - consec_len:
            continue
        others = random.sample(remaining_pool, block_size - consec_len)
        tokens = consec + others
        random.shuffle(tokens)
        x = torch.tensor(tokens, dtype=torch.long, device=device)
        vals = x.sort().values
        sep = torch.tensor([vocab_n], dtype=torch.long, device=device)
        idx = torch.cat([x, sep, vals]).unsqueeze(0)
        sorted_vals = vals.tolist()
        for j in range(len(sorted_vals) - consec_len + 1):
            if sorted_vals[j:j + consec_len] == consec:
                return idx, i_start


@torch.no_grad()
def main():
    print(f"Loading model from {ARGS.ckpt}")
    model = load_model_from_checkpoint(ARGS.ckpt)
    block_size = model.config.block_size
    vocab_n = model.config.vocab_size - 1
    output_path = ARGS.output or os.path.join(
        os.path.dirname(ARGS.ckpt), "..", "plots", "consecutive_attention_grid.png"
    )
    print(f"  block_size={block_size}, vocab_n={vocab_n}, device={DEVICE}")

    ncols = 5
    nrows = (N_SAMPLES + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.2 * nrows),
                             sharex=False, sharey=True)
    axes_flat = axes.flatten()

    for ax in axes_flat:
        ax.set_visible(False)

    sample_i = 0
    while sample_i < N_SAMPLES:
        idx, i_start = generate_consecutive_batch(block_size, vocab_n, CONSEC_LEN, DEVICE)
        target_val = i_start + 1

        sorted_part = idx[0, block_size + 1: 2 * block_size + 1]
        positions = (sorted_part == target_val).nonzero(as_tuple=True)[0]
        if len(positions) == 0:
            continue
        sorted_pos = positions[0].item()
        query_pos = block_size + 1 + sorted_pos

        all_probs = get_all_layer_probs(model, idx)
        unsorted_tokens = idx[0, :block_size].cpu().numpy()

        ax = axes_flat[sample_i]
        ax.set_visible(True)

        for layer, color, marker in [(0, "#2166ac", "o"), (1, "#b2182b", "s")]:
            attn = all_probs[layer][0].mean(dim=0)
            attn_to_unsorted = attn[query_pos, :block_size].cpu().numpy()

            nontrivial = attn_to_unsorted > THRESHOLD
            if not nontrivial.any():
                continue

            tok_vals = unsorted_tokens[nontrivial]
            scores = attn_to_unsorted[nontrivial]
            offsets = tok_vals.astype(int) - int(target_val)

            ax.bar(offsets + (-0.15 if layer == 0 else 0.15), scores,
                   width=0.3, color=color, alpha=0.7,
                   label=f"L{layer+1}" if sample_i == 0 else None)

        ax.axvline(x=1, color="red", linestyle="--", linewidth=0.6, alpha=0.4)
        ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.6, alpha=0.4)

        ax.set_title(f"i={i_start}", fontsize=9, pad=3)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.15, linestyle=":")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if sample_i % ncols == 0:
            ax.set_ylabel("Attn score", fontsize=8)
        if sample_i >= (nrows - 1) * ncols:
            ax.set_xlabel("Offset from i+1", fontsize=8)

        if (sample_i + 1) % 10 == 0:
            print(f"  Processed {sample_i + 1}/{N_SAMPLES}")
        sample_i += 1

    fig.legend(["Layer 1", "Layer 2"], loc="upper right", fontsize=10,
               bbox_to_anchor=(0.99, 0.99))
    fig.suptitle(
        f"Per-sample attention at sorted position of i+1  "
        f"({CONSEC_LEN} consecutive numbers)\n"
        f"k={block_size}, N={vocab_n}, threshold>{THRESHOLD}  |  "
        f"blue=Layer 1, red=Layer 2, dashed red=i+2 (correct next)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
