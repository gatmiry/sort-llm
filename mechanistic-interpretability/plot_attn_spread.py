#!/usr/bin/env python3
"""
Measure the average numerical distance (|query_value - key_value|) between
the sorted query token and the unsorted key tokens with non-trivial attention,
for both Layer 1 and Layer 2.

Plot A: Avg value distance vs sorted query position (fixed threshold).
Plot B: Avg value distance vs attention threshold (averaged over positions).
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
parser.add_argument("--n-trials", type=int, default=800)
ARGS = parser.parse_args()


@torch.no_grad()
def main():
    model = load_model_from_checkpoint(ARGS.ckpt)
    block_size = model.config.block_size
    vocab_n = model.config.vocab_size - 1
    n_layers = model.config.n_layers

    output_path = ARGS.output or os.path.join(
        os.path.dirname(ARGS.ckpt), "..", "plots", "attn_value_distance.png"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    enable_attention_storage(model)

    n_trials = ARGS.n_trials
    n_sorted_positions = block_size

    fixed_threshold = 0.04
    thresholds = [0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3]

    # Plot A accumulators: per sorted position, per layer
    vdist_by_pos = {layer: [[] for _ in range(n_sorted_positions)] for layer in range(n_layers)}
    n_attended_by_pos = {layer: [[] for _ in range(n_sorted_positions)] for layer in range(n_layers)}

    # Plot B accumulators: per threshold, per layer
    vdist_by_thresh = {layer: [[] for _ in range(len(thresholds))] for layer in range(n_layers)}
    n_attended_by_thresh = {layer: [[] for _ in range(len(thresholds))] for layer in range(n_layers)}

    for trial in range(n_trials):
        idx = get_batch(1, block_size, DEVICE, vocab_n=vocab_n)
        model(idx, block_size=block_size)

        tokens = idx[0].cpu().numpy()
        unsorted_values = tokens[:block_size]
        sorted_values = tokens[block_size + 1:]

        for layer in range(n_layers):
            attn = model.transformer.h[layer].attn.attn.cpu().numpy()  # (T, T)

            for p in range(n_sorted_positions):
                query_pos = block_size + p
                query_value = sorted_values[p] if p < len(sorted_values) else tokens[query_pos]

                unsorted_attn = attn[query_pos, :block_size]

                # Plot A: fixed threshold
                attended_mask = unsorted_attn > fixed_threshold
                if attended_mask.any():
                    attended_vals = unsorted_values[attended_mask]
                    dists = np.abs(attended_vals.astype(float) - float(query_value))
                    avg_dist = np.mean(dists)
                    vdist_by_pos[layer][p].append(avg_dist)
                    n_attended_by_pos[layer][p].append(int(attended_mask.sum()))

                # Plot B: sweep thresholds
                for ti, thresh in enumerate(thresholds):
                    mask_t = unsorted_attn > thresh
                    if mask_t.any():
                        vals_t = unsorted_values[mask_t]
                        dists_t = np.abs(vals_t.astype(float) - float(query_value))
                        vdist_by_thresh[layer][ti].append(np.mean(dists_t))
                        n_attended_by_thresh[layer][ti].append(int(mask_t.sum()))

        if (trial + 1) % 200 == 0:
            print(f"  {trial+1}/{n_trials}")

    # ========== PLOTTING ==========
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    layer_colors = {0: "#2166ac", 1: "#b2182b"}
    layer_names = {0: "Layer 1", 1: "Layer 2"}

    # --- Plot A (top-left): Avg value distance vs sorted position ---
    ax = axes[0, 0]
    for layer in range(n_layers):
        means, stds, positions = [], [], []
        for p in range(n_sorted_positions):
            if vdist_by_pos[layer][p]:
                means.append(np.mean(vdist_by_pos[layer][p]))
                stds.append(np.std(vdist_by_pos[layer][p]) / np.sqrt(len(vdist_by_pos[layer][p])))
                positions.append(p)
        ax.plot(positions, means, color=layer_colors[layer], linewidth=2,
                label=layer_names[layer])
        ax.fill_between(positions,
                        np.array(means) - np.array(stds),
                        np.array(means) + np.array(stds),
                        color=layer_colors[layer], alpha=0.15)
    ax.set_xlabel("Sorted query position index", fontsize=11)
    ax.set_ylabel("Avg |query_value − key_value|", fontsize=11)
    ax.set_title(f"Avg numerical distance: sorted query → attended unsorted keys\n"
                 f"(threshold > {fixed_threshold})",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # --- Plot A2 (top-right): Avg # attended keys vs sorted position ---
    ax = axes[0, 1]
    for layer in range(n_layers):
        means, positions = [], []
        for p in range(n_sorted_positions):
            if n_attended_by_pos[layer][p]:
                means.append(np.mean(n_attended_by_pos[layer][p]))
                positions.append(p)
        ax.plot(positions, means, color=layer_colors[layer], linewidth=2,
                label=layer_names[layer])
    ax.set_xlabel("Sorted query position index", fontsize=11)
    ax.set_ylabel("Avg number of attended unsorted keys", fontsize=11)
    ax.set_title(f"Number of unsorted keys with attn > {fixed_threshold}\nvs sorted position",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # --- Plot B (bottom-left): Avg value distance vs threshold ---
    ax = axes[1, 0]
    for layer in range(n_layers):
        means, stds, valid_thresh = [], [], []
        for ti, thresh in enumerate(thresholds):
            if vdist_by_thresh[layer][ti]:
                means.append(np.mean(vdist_by_thresh[layer][ti]))
                stds.append(np.std(vdist_by_thresh[layer][ti]) / np.sqrt(len(vdist_by_thresh[layer][ti])))
                valid_thresh.append(thresh)
        ax.plot(valid_thresh, means, "o-", color=layer_colors[layer], linewidth=2,
                markersize=5, label=layer_names[layer])
        ax.fill_between(valid_thresh,
                        np.array(means) - np.array(stds),
                        np.array(means) + np.array(stds),
                        color=layer_colors[layer], alpha=0.15)
    ax.set_xlabel("Attention threshold", fontsize=11)
    ax.set_ylabel("Avg |query_value − key_value|", fontsize=11)
    ax.set_title("Avg numerical distance vs attention threshold\n(averaged over all sorted positions)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # --- Plot B2 (bottom-right): Avg # attended keys vs threshold ---
    ax = axes[1, 1]
    for layer in range(n_layers):
        means, valid_thresh = [], []
        for ti, thresh in enumerate(thresholds):
            if n_attended_by_thresh[layer][ti]:
                means.append(np.mean(n_attended_by_thresh[layer][ti]))
                valid_thresh.append(thresh)
        ax.plot(valid_thresh, means, "o-", color=layer_colors[layer], linewidth=2,
                markersize=5, label=layer_names[layer])
    ax.set_xlabel("Attention threshold", fontsize=11)
    ax.set_ylabel("Avg number of attended unsorted keys", fontsize=11)
    ax.set_title("Number of attended keys vs threshold\n(averaged over all sorted positions)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle(
        f"Numerical distance between sorted query and attended unsorted keys\n"
        f"k={block_size}, N={vocab_n}, {os.path.basename(ARGS.ckpt)}  |  {n_trials} trials",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    for layer in range(n_layers):
        all_dists = [d for pos_list in vdist_by_pos[layer] for d in pos_list]
        all_counts = [c for pos_list in n_attended_by_pos[layer] for c in pos_list]
        if all_dists:
            print(f"{layer_names[layer]} (threshold={fixed_threshold}):")
            print(f"  Avg value distance: {np.mean(all_dists):.2f} ± {np.std(all_dists):.2f}")
            print(f"  Avg # attended keys: {np.mean(all_counts):.1f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
