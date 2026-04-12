#!/usr/bin/env python3
"""
For the k32_N256 model: generate inputs containing 6 consecutive numbers
i, i+1, ..., i+5 in the sorted output. At the sorted position of i+1,
extract attention scores to unsorted tokens with score > 0.05.
Plot all 50 samples overlaid, one subplot per layer.
X-axis: token value offset relative to i+1.
Y-axis: attention score.
"""

import os
import sys
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "sort-llm", "sortgpt_toolkit"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from model import DEVICE, load_model_from_checkpoint
from plot_heatmaps import get_all_layer_probs, get_batch_simple

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", default="/mnt/task_runtime/new-grid/k32_N256/checkpoints/std0p01_iseed1__ckpt100000.pt")
parser.add_argument("--output", default=None)
parser.add_argument("--block-size", type=int, default=None)
parser.add_argument("--vocab-n", type=int, default=None)
parser.add_argument("--n-samples", type=int, default=50)
parser.add_argument("--consec-len", type=int, default=6)
parser.add_argument("--threshold", type=float, default=0.05)
_ARGS = parser.parse_args()

CKPT = _ARGS.ckpt
N_SAMPLES = _ARGS.n_samples
CONSEC_LEN = _ARGS.consec_len
THRESHOLD = _ARGS.threshold


def generate_consecutive_batch(block_size, vocab_n, consec_len, device):
    """Generate a single sample guaranteed to have `consec_len` consecutive
    numbers in its sorted output. Returns (idx, i_value) where i_value is
    the start of the consecutive run."""
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


def find_sorted_position(idx, block_size, target_value):
    """Find the position index within the sorted region for a given value."""
    sorted_part = idx[0, block_size + 1: 2 * block_size + 1]
    positions = (sorted_part == target_value).nonzero(as_tuple=True)[0]
    if len(positions) == 0:
        return None
    return positions[0].item()


@torch.no_grad()
def main():
    print(f"Loading model from {CKPT}")
    model = load_model_from_checkpoint(CKPT)
    BLOCK_SIZE = _ARGS.block_size or model.config.block_size
    VOCAB_N = _ARGS.vocab_n or (model.config.vocab_size - 1)
    OUTPUT = _ARGS.output or os.path.join(os.path.dirname(CKPT), "..", "plots", "consecutive_attention.png")
    print(f"  block_size={BLOCK_SIZE}, vocab_n={VOCAB_N}, device={DEVICE}")

    layer_data = {0: [], 1: []}

    for sample_idx in range(N_SAMPLES):
        idx, i_start = generate_consecutive_batch(BLOCK_SIZE, VOCAB_N, CONSEC_LEN, DEVICE)
        target_val = i_start + 1

        sorted_pos = find_sorted_position(idx, BLOCK_SIZE, target_val)
        if sorted_pos is None:
            print(f"  sample {sample_idx}: could not find {target_val} in sorted — skipping")
            continue

        query_pos = BLOCK_SIZE + 1 + sorted_pos

        all_probs = get_all_layer_probs(model, idx)

        unsorted_tokens = idx[0, :BLOCK_SIZE].cpu().numpy()

        for layer in [0, 1]:
            attn = all_probs[layer][0].mean(dim=0)
            attn_to_unsorted = attn[query_pos, :BLOCK_SIZE].cpu().numpy()

            nontrivial_mask = attn_to_unsorted > THRESHOLD
            if not nontrivial_mask.any():
                continue

            token_values = unsorted_tokens[nontrivial_mask]
            scores = attn_to_unsorted[nontrivial_mask]
            offsets = token_values.astype(int) - int(target_val)

            layer_data[layer].append((offsets, scores, sample_idx))

        if (sample_idx + 1) % 10 == 0:
            print(f"  Processed {sample_idx + 1}/{N_SAMPLES} samples")

    print(f"Collected data: layer0={len(layer_data[0])} samples, layer1={len(layer_data[1])} samples")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    cmap = plt.cm.tab20(np.linspace(0, 1, N_SAMPLES))

    for li, ax in enumerate(axes):
        for offsets, scores, si in layer_data[li]:
            ax.scatter(offsets, scores, color=cmap[si % len(cmap)],
                       alpha=0.5, s=25, edgecolors="none")
            ax.plot(offsets, scores, color=cmap[si % len(cmap)],
                    alpha=0.15, linewidth=0.5)

        ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5,
                   label="i+1 (query token)")
        ax.axvline(x=1, color="red", linestyle="--", linewidth=0.8, alpha=0.5,
                   label="i+2 (correct next)")
        ax.set_xlabel("Token value offset from i+1", fontsize=11)
        ax.set_title(f"Layer {li + 1} attention", fontsize=13, fontweight="bold")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.2, linestyle=":")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Attention score", fontsize=11)

    fig.suptitle(
        f"Attention at sorted position of i+1 (6 consecutive numbers i..i+5)\n"
        f"k={BLOCK_SIZE}, N={VOCAB_N}, {N_SAMPLES} samples, threshold>{THRESHOLD}",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {OUTPUT}")


if __name__ == "__main__":
    main()
