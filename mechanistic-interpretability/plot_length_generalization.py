#!/usr/bin/env python3
"""
Measure how attention spread and accuracy change as sequence length
increases beyond the training length (block_size=32).

For each length k:
  - Generate random inputs of length k (no duplicates, same distribution)
  - Count avg number of unsorted keys with attn > threshold per sorted query
  - Measure per-token accuracy

Plot both L1 and L2 attention counts + accuracy vs k.
"""

import os, sys, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "sort-llm", "sortgpt_toolkit"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from model import DEVICE, load_model_from_checkpoint, get_batch
from intervene import enable_attention_storage

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", required=True)
parser.add_argument("--output", default=None)
parser.add_argument("--n-trials", type=int, default=200)
ARGS = parser.parse_args()


@torch.no_grad()
def main():
    model = load_model_from_checkpoint(ARGS.ckpt)
    train_block_size = model.config.block_size
    vocab_n = model.config.vocab_size - 1

    max_k = min(256, vocab_n)
    max_seq = 2 * max_k + 1
    if max_seq > model.config.max_seq_len:
        model = load_model_from_checkpoint(ARGS.ckpt, extended_max_seq_len=max_seq)
    n_layers = model.config.n_layers

    output_path = ARGS.output or os.path.join(
        os.path.dirname(ARGS.ckpt), "..", "plots", "length_generalization.png"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    enable_attention_storage(model)

    lengths = list(range(32, max_k + 1, 8))
    if lengths[-1] != max_k:
        lengths.append(max_k)

    n_trials = ARGS.n_trials
    threshold = 0.02

    # Accumulators
    avg_n_attended = {layer: [] for layer in range(n_layers)}
    avg_n_attended_se = {layer: [] for layer in range(n_layers)}
    per_token_acc = []
    per_token_acc_se = []

    for k in lengths:
        print(f"  k={k} ...", end=" ", flush=True)
        trial_counts = {layer: [] for layer in range(n_layers)}
        trial_accs = []

        for _ in range(n_trials):
            idx = get_batch(1, k, DEVICE, vocab_n=vocab_n)
            logits, loss = model(idx, block_size=k, return_full_logits=True)

            # Accuracy: predictions at sorted positions
            preds = torch.argmax(logits[0, k:2*k, :], dim=1)
            targets = idx[0, k+1:]
            correct = (preds == targets).float().mean().item()
            trial_accs.append(correct)

            # Attention counts
            for layer in range(n_layers):
                attn = model.transformer.h[layer].attn.attn.cpu().numpy()
                counts = []
                for p in range(k):
                    query_pos = k + p
                    unsorted_attn = attn[query_pos, :k]
                    n_above = int(np.sum(unsorted_attn > threshold))
                    counts.append(n_above)
                trial_counts[layer].append(np.mean(counts))

        for layer in range(n_layers):
            m = np.mean(trial_counts[layer])
            se = np.std(trial_counts[layer]) / np.sqrt(len(trial_counts[layer]))
            avg_n_attended[layer].append(m)
            avg_n_attended_se[layer].append(se)

        acc_m = np.mean(trial_accs)
        acc_se = np.std(trial_accs) / np.sqrt(len(trial_accs))
        per_token_acc.append(acc_m)
        per_token_acc_se.append(acc_se)

        print(f"L1={avg_n_attended[0][-1]:.1f}  L2={avg_n_attended[1][-1]:.1f}  "
              f"acc={acc_m:.3f}")

    # ========== PLOTTING ==========
    fig, ax1 = plt.subplots(figsize=(14, 7))

    layer_colors = {0: "#2166ac", 1: "#b2182b"}
    layer_names = {0: "Layer 1", 1: "Layer 2"}

    for layer in range(n_layers):
        means = np.array(avg_n_attended[layer])
        ses = np.array(avg_n_attended_se[layer])
        ax1.plot(lengths, means, "o-", color=layer_colors[layer], linewidth=2,
                 markersize=5, label=f"{layer_names[layer]}: avg # keys with attn > {threshold}")
        ax1.fill_between(lengths, means - ses, means + ses,
                         color=layer_colors[layer], alpha=0.12)

    ax1.set_xlabel("Sequence length k (unsorted = sorted length)", fontsize=12)
    ax1.set_ylabel(f"Avg # unsorted keys with attn > {threshold}", fontsize=12)
    ax1.axvline(train_block_size, color="gray", linewidth=1, linestyle="--", alpha=0.5,
                label=f"Training length (k={train_block_size})")

    # Accuracy on secondary y-axis
    ax2 = ax1.twinx()
    acc_arr = np.array(per_token_acc)
    acc_se_arr = np.array(per_token_acc_se)
    ax2.plot(lengths, acc_arr, "s-", color="#1b7837", linewidth=2, markersize=5,
             label="Per-token accuracy")
    ax2.fill_between(lengths, acc_arr - acc_se_arr, acc_arr + acc_se_arr,
                     color="#1b7837", alpha=0.12)
    ax2.set_ylabel("Per-token accuracy", fontsize=12, color="#1b7837")
    ax2.tick_params(axis="y", labelcolor="#1b7837")
    ax2.set_ylim(-0.05, 1.05)

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc="center left")

    ax1.grid(True, alpha=0.15)
    ax1.set_title(
        f"Length generalization: attention spread and accuracy\n"
        f"Trained at k={train_block_size}, N={vocab_n}, {os.path.basename(ARGS.ckpt)}  |  "
        f"{n_trials} trials per length",
        fontsize=13, fontweight="bold",
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
