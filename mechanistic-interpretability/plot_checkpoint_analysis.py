#!/usr/bin/env python3
"""
Produce the same analysis plots as plots_V64_B32_LR1e-02_MI10000_E64_H1_L2_ckpt10000
for a single sortgpt_toolkit checkpoint.

Generates:
  baseline_accuracy.png, baseline_conditional_accuracy.png,
  ablation_accuracy.png, ablation_per_position.png, ablation_conditional_accuracy.png,
  compare_cinclogits_layer{0,1}.png,
  compare_intensity_layer{0,1}.png,
  compare_intensity_layer{0,1}_ub{10,15}.png

Usage:
    CUDA_VISIBLE_DEVICES=X python plot_checkpoint_analysis.py \
        --ckpt /path/to/checkpoint.pt --output-dir /path/to/plots
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "sort-llm", "sortgpt_toolkit"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from model import DEVICE, load_model_from_checkpoint
from intervene import (
    enable_attention_storage,
    compute_baseline,
    compute_ablation,
    compute_cinclogits,
    compute_intensity,
)


def plot_baseline(bl, plot_dir, tag):
    fig, ax = plt.subplots(figsize=(4, 4))
    val = bl["full_seq_acc"]
    bar = ax.bar([0], [val], 0.4, color="#e6850e")
    ax.text(0, val + 0.01, f"{val:.3f}", ha="center", va="bottom",
            fontsize=10, fontweight="bold")
    ax.set_xticks([0])
    ax.set_xticklabels(["Model"], fontsize=11)
    ax.set_ylabel("Full-sequence accuracy", fontsize=12)
    ax.set_title(f"Baseline accuracy (500 trials)\n{tag}", fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.2, linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, min(1.15, val * 1.2 + 0.05))
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "baseline_accuracy.png"), dpi=300, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(6, 4))
    ca = bl["cond_acc"]
    ce = bl["cond_eligible"]
    pos = np.arange(len(ca))
    valid = ce >= 10
    ax.plot(pos[valid], ca[valid], marker="s", markersize=3,
            linewidth=1.2, label="Model", color="#e6850e")
    ax.set_xlabel("Output position", fontsize=10)
    ax.set_ylabel("Conditional per-token accuracy", fontsize=10)
    ax.set_title(f"Per-token accuracy (given correct prefix) — baseline\n{tag}",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "baseline_conditional_accuracy.png"),
                dpi=300, bbox_inches="tight")
    plt.close()


def plot_ablation(abl_data, plot_dir, tag):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    labels = ["Skip Layer 0", "Skip Layer 1"]
    x = np.arange(len(labels))
    vals = [float(abl_data[0]["full_seq_acc"]), float(abl_data[1]["full_seq_acc"])]
    bars = ax.bar(x, vals, 0.4, color=["#6a3d9a", "#e6850e"])
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + 0.01,
                f"{h:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Full-sequence accuracy", fontsize=12)
    ax.set_title(f"Accuracy with attention layer removed (500 trials)\n{tag}",
                 fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.2, linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, min(1.15, max(vals) * 1.25 + 0.05))
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "ablation_accuracy.png"), dpi=300, bbox_inches="tight")
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for i, layer in enumerate([0, 1]):
        ax = axes[i]
        ppa = abl_data[layer]["per_pos_acc"]
        pos = np.arange(len(ppa))
        ax.plot(pos, ppa, marker="o", markersize=3, linewidth=1.2, color="#6a3d9a")
        ax.set_xlabel("Output position", fontsize=10)
        ax.set_title(f"Skip Layer {layer}", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("Per-position accuracy", fontsize=10)
    fig.suptitle(f"Per-position accuracy with attention removed (500 trials)\n{tag}",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "ablation_per_position.png"), dpi=300, bbox_inches="tight")
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for i, layer in enumerate([0, 1]):
        ax = axes[i]
        ca = abl_data[layer]["cond_acc"]
        ce = abl_data[layer]["cond_eligible"]
        pos = np.arange(len(ca))
        valid = ce >= 10
        if valid.any():
            ax.plot(pos[valid], ca[valid], marker="o", markersize=3,
                    linewidth=1.2, color="#6a3d9a")
            if not valid.all():
                cutoff = np.where(~valid)[0][0]
                ax.axvline(x=cutoff - 0.5, color="#6a3d9a", linestyle=":", alpha=0.5)
        ax.set_xlabel("Output position", fontsize=10)
        ax.set_title(f"Skip Layer {layer}", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("Conditional per-token accuracy", fontsize=10)
    fig.suptitle(f"Per-token accuracy (given prefix correct) with attention removed\n{tag}",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "ablation_conditional_accuracy.png"),
                dpi=300, bbox_inches="tight")
    plt.close()


def plot_cinclogits(cl_ic, icl_ic, plot_dir, layer, tag):
    frac = np.mean(cl_ic + icl_ic)
    eps = 1e-10
    corr = np.sum(cl_ic) / (np.sum(cl_ic + icl_ic) + eps)

    fig, ax = plt.subplots(figsize=(5, 4.2))
    bw = 0.4
    x = np.array([0, 1])
    b1 = ax.bar(x[0], frac, bw, color="#e6850e")
    b2 = ax.bar(x[1], corr, bw, color="#e6850e")
    for bar in [b1, b2]:
        for b in bar:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, h + 0.008,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(["Fraction of\nincorrect scores",
                        "Logit correction ratio\namong incorrect scores"], fontsize=11)
    ax.set_ylabel("Fraction", fontsize=12)
    ax.set_title(f"Incorrect scores & logit correction (Layer {layer})\n{tag}",
                 fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.2, linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ymax = max(frac, corr)
    ax.set_ylim(0, ymax * 1.25 if ymax > 0 else 1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, f"compare_cinclogits_layer{layer}.png"),
                dpi=300, bbox_inches="tight")
    plt.close()


def plot_intensity(intensities, rates, layer, plot_dir, tag, ub_suffix=""):
    plt.figure(figsize=(3.5, 2.8))
    plt.plot(intensities, rates, marker="o", linewidth=1.5,
             markersize=5, label="model", color="#1f77b4")
    plt.xlabel("Intervention Intensity", fontsize=9)
    plt.ylabel("Success Probability", fontsize=9)
    title = f"Robustness to Attention Intervention (Layer {layer})"
    if ub_suffix:
        title += f"  [ub={ub_suffix}]"
    title += f"\n{tag}"
    plt.title(title, fontsize=10)
    plt.legend(fontsize=7, loc="lower left")
    plt.grid(True, alpha=0.3)
    if len(intensities) > 2:
        plt.xticks(list(intensities[::2]), fontsize=8)
    plt.yticks(fontsize=8)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    fname = f"compare_intensity_layer{layer}"
    if ub_suffix:
        fname += f"_ub{ub_suffix}"
    plt.savefig(os.path.join(plot_dir, f"{fname}.png"), dpi=300, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Checkpoint .pt file")
    parser.add_argument("--output-dir", required=True, help="Directory to save plots")
    parser.add_argument("--tag", default="", help="Title tag for plots")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading checkpoint: {args.ckpt}", flush=True)
    model = load_model_from_checkpoint(args.ckpt)
    block_size = model.config.block_size
    vocab_n = model.config.vocab_size - 1
    tag = args.tag or f"k={block_size} N={vocab_n}"
    print(f"  block_size={block_size}, vocab_n={vocab_n}, device={DEVICE}", flush=True)

    enable_attention_storage(model)

    # 1. Baseline
    print("Computing baseline...", flush=True)
    bl = compute_baseline(model, block_size, vocab_n, DEVICE, num_trials=500)
    print(f"  full_seq_acc={bl['full_seq_acc']:.4f}", flush=True)
    plot_baseline(bl, args.output_dir, tag)
    print("  Saved baseline plots", flush=True)

    # 2. Ablation (skip layer 0, skip layer 1)
    print("Computing ablation...", flush=True)
    abl = {}
    for layer in range(2):
        abl[layer] = compute_ablation(model, block_size, vocab_n, DEVICE,
                                      skip_layer=layer, num_trials=500)
        print(f"  skip_layer={layer}: full_seq_acc={abl[layer]['full_seq_acc']:.4f}", flush=True)
    plot_ablation(abl, args.output_dir, tag)
    print("  Saved ablation plots", flush=True)

    # 3. Cinclogits
    print("Computing cinclogits...", flush=True)
    for layer in range(2):
        cl_ic, icl_ic = compute_cinclogits(model, block_size, vocab_n, DEVICE,
                                           attn_layer=layer, num_tries=100)
        plot_cinclogits(cl_ic, icl_ic, args.output_dir, layer, tag)
        print(f"  Layer {layer}: done", flush=True)
    print("  Saved cinclogits plots", flush=True)

    # 4. Intensity sweeps
    ub_configs = [
        (5, ""),
        (10, "10"),
        (15, "15"),
    ]
    for layer in range(2):
        for ub_val, ub_suffix in ub_configs:
            print(f"Computing intensity layer={layer} ub={ub_val}...", flush=True)
            intensities, rates, counts = compute_intensity(
                model, block_size, vocab_n, DEVICE, attn_layer=layer,
                unsorted_lb=ub_val, unsorted_ub=ub_val,
            )
            plot_intensity(intensities, rates, layer, args.output_dir, tag, ub_suffix)
            print(f"  Done", flush=True)
    print("  Saved intensity plots", flush=True)

    print(f"\nAll plots saved to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
