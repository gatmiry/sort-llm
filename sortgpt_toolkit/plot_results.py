#!/usr/bin/env python3
"""
Post-hoc evaluation and plotting for a completed (or in-progress) run.

Evaluates length generalization and ablation at each checkpoint,
then generates combined plots.

Usage:
    CUDA_VISIBLE_DEVICES=X python plot_results.py \\
        --run-dir ./my_run \\
        --seeds 1501 1502 1503 1504 1505 \\
        --init-stds 0.02 0.02 0.02 0.02 0.02 \\
        --checkpoint-every 20000
"""

import argparse
import gc
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch

from model import DEVICE, float_token, load_model_from_checkpoint
from evaluate import (
    evaluate_ablation, evaluate_length_generalization,
    evaluate_at_length, ablate_attention, evaluate_token_accuracy,
    EVAL_LENGTHS,
)

VOCAB_N = 256
BLOCK_SIZE = 16
EXTENDED_MAX_SEQ_LEN = 2 * VOCAB_N + 1


def main():
    parser = argparse.ArgumentParser(description="Evaluate and plot results from a run")
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--init-stds", type=float, nargs="+", required=True)
    parser.add_argument("--checkpoint-every", type=int, default=20_000)
    parser.add_argument("--max-iters", type=int, default=None,
                        help="Max iteration to look for (auto-detected if omitted)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    ckpt_dir = run_dir / "checkpoints"
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    seeds = args.seeds
    init_stds = args.init_stds
    assert len(seeds) == len(init_stds)
    std_by_seed = dict(zip(seeds, init_stds))

    # Auto-detect max iteration from available checkpoints
    if args.max_iters:
        max_iter = args.max_iters
    else:
        all_ckpts = sorted(ckpt_dir.glob("*.pt"))
        max_iter = 0
        for f in all_ckpts:
            import re
            m = re.search(r'ckpt(\d+)', f.stem)
            if m:
                max_iter = max(max_iter, int(m.group(1)))
        if max_iter == 0:
            print("No checkpoints found!")
            return

    checkpoint_iters = list(range(args.checkpoint_every, max_iter + 1, args.checkpoint_every))
    print(f"=== Evaluating {len(seeds)} seeds × {len(checkpoint_iters)} checkpoints ===")
    print(f"    Seeds: {seeds}")
    print(f"    Stds:  {init_stds}")
    print(f"    Checkpoints: {[f'{x//1000}k' for x in checkpoint_iters]}")

    # Collect data: {seed: {iter: {ablation + lengthgen}}}
    all_data = {s: {} for s in seeds}

    for seed in seeds:
        tag = f"std{float_token(std_by_seed[seed])}_iseed{seed}"
        for iteration in checkpoint_iters:
            ckpt_path = ckpt_dir / f"{tag}__ckpt{iteration}.pt"
            if not ckpt_path.exists():
                print(f"  [SKIP] seed={seed} ckpt={iteration//1000}k — not found")
                continue

            print(f"  seed={seed} @ {iteration//1000}k...")
            model = load_model_from_checkpoint(ckpt_path,
                                                extended_max_seq_len=EXTENDED_MAX_SEQ_LEN)

            # Length gen
            accs = evaluate_length_generalization(model, VOCAB_N)
            L16i, L64i, L256i = EVAL_LENGTHS.index(16), EVAL_LENGTHS.index(64), EVAL_LENGTHS.index(256)
            print(f"    L16={accs[L16i]:.3f}  L64={accs[L64i]:.3f}  L256={accs[L256i]:.3f}")

            # Ablation
            ablation = evaluate_ablation(model, BLOCK_SIZE, VOCAB_N)
            print(f"    full={ablation['full_tok']:.4f}  no1={ablation['no_attn1_tok']:.4f}  "
                  f"no2={ablation['no_attn2_tok']:.4f}")

            all_data[seed][iteration] = {**ablation, "lengthgen": accs}

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    # ═══════════════════════════════════════════════════════════════════════════
    # Plot 1: Length gen — one subplot per checkpoint, all seeds overlaid
    # ═══════════════════════════════════════════════════════════════════════════

    n_ckpts = len(checkpoint_iters)
    fig, axes = plt.subplots(1, n_ckpts, figsize=(6 * n_ckpts, 6), sharey=True)
    if n_ckpts == 1:
        axes = [axes]

    for ci, iteration in enumerate(checkpoint_iters):
        ax = axes[ci]
        for seed in seeds:
            if iteration not in all_data[seed]:
                continue
            d = all_data[seed][iteration]
            no2 = d["no_attn2_tok"]
            color = "green" if no2 > 0.95 else "red"
            ax.plot(EVAL_LENGTHS, d["lengthgen"], "-o", color=color, linewidth=1.8,
                    markersize=3, alpha=0.8, label=f"s{seed} (no2={no2:.2f})")
        ax.axvline(x=BLOCK_SIZE, color="gray", linestyle="--", linewidth=1, alpha=0.5)
        ax.set_title(f"{iteration//1000}k steps", fontsize=12, fontweight="bold")
        ax.set_xlabel("Sequence Length", fontsize=10)
        if ci == 0:
            ax.set_ylabel("Token Accuracy", fontsize=11)
        ax.set_ylim(-0.02, 1.05)
        ax.set_xticks(EVAL_LENGTHS[::4])
        ax.set_xticklabels([str(x) for x in EVAL_LENGTHS[::4]], fontsize=7, rotation=45)
        ax.legend(fontsize=7, loc="lower left")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Length Generalization — GREEN=no_attn2>0.95, RED<0.95",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = plots_dir / "lengthgen_all_checkpoints.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {path}")

    # ═══════════════════════════════════════════════════════════════════════════
    # Plot 2: Per-seed progression — one subplot per seed, checkpoints overlaid
    # ═══════════════════════════════════════════════════════════════════════════

    n_seeds = len(seeds)
    fig, axes = plt.subplots(1, n_seeds, figsize=(6 * n_seeds, 6), sharey=True)
    if n_seeds == 1:
        axes = [axes]
    ckpt_colors = plt.cm.viridis(np.linspace(0.15, 0.95, n_ckpts))

    for si, seed in enumerate(seeds):
        ax = axes[si]
        for ci, iteration in enumerate(checkpoint_iters):
            if iteration not in all_data[seed]:
                continue
            ax.plot(EVAL_LENGTHS, all_data[seed][iteration]["lengthgen"],
                    "-o", color=ckpt_colors[ci], linewidth=1.5, markersize=3,
                    alpha=0.85, label=f"{iteration//1000}k")
        final_iter = max(all_data[seed].keys()) if all_data[seed] else None
        border = "green" if (final_iter and all_data[seed][final_iter]["no_attn2_tok"] > 0.95) else "red"
        ax.axvline(x=BLOCK_SIZE, color="gray", linestyle="--", linewidth=1, alpha=0.5)
        ax.set_title(f"seed={seed}", fontsize=12, fontweight="bold", color=border)
        ax.set_xlabel("Sequence Length", fontsize=10)
        if si == 0:
            ax.set_ylabel("Token Accuracy", fontsize=11)
        ax.set_ylim(-0.02, 1.05)
        ax.set_xticks(EVAL_LENGTHS[::4])
        ax.set_xticklabels([str(x) for x in EVAL_LENGTHS[::4]], fontsize=7, rotation=45)
        ax.legend(fontsize=8, loc="lower left", title="Checkpoint")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)
        for spine in ax.spines.values():
            spine.set_edgecolor(border)
            spine.set_linewidth(2)

    fig.suptitle("Length Gen Progression — GREEN=no_attn2>0.95 at final ckpt",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = plots_dir / "lengthgen_per_seed_progression.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    # ═══════════════════════════════════════════════════════════════════════════
    # Plot 3: Ablation bar plots at each checkpoint
    # ═══════════════════════════════════════════════════════════════════════════

    BAR_COLORS = ['#4C72B0', '#DD8452', '#55A868']
    BAR_LABELS = ['Full model', 'Attn-1 removed', 'Attn-2 removed']
    CONDITIONS = ['full_tok', 'no_attn1_tok', 'no_attn2_tok']

    for iteration in checkpoint_iters:
        seeds_with_data = [s for s in seeds if iteration in all_data[s]]
        if not seeds_with_data:
            continue

        n = len(seeds_with_data)
        fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 5), squeeze=False)

        for col, seed in enumerate(seeds_with_data):
            ax = axes[0][col]
            d = all_data[seed][iteration]
            no2 = d["no_attn2_tok"]
            border = "green" if no2 > 0.95 else "red"

            for i, (cond, color, lbl) in enumerate(zip(CONDITIONS, BAR_COLORS, BAR_LABELS)):
                val = d[cond]
                ax.bar(i, val, width=0.55, color=color, edgecolor='white', linewidth=0.8, label=lbl)
                ax.text(i, val + 0.012, f"{val:.3f}", ha='center', va='bottom',
                        fontsize=9, fontweight='bold')

            ax.set_title(f"seed={seed}", fontsize=11, color=border, fontweight='bold')
            ax.set_xticks(range(3))
            ax.set_xticklabels(['Full', 'No Attn-1', 'No Attn-2'], fontsize=9)
            ax.set_ylim(0, 1.12)
            ax.set_ylabel('Token Accuracy', fontsize=9)
            ax.spines[['top', 'right']].set_visible(False)
            ax.grid(axis='y', alpha=0.35, linestyle='--')
            for spine in ax.spines.values():
                spine.set_edgecolor(border)
                spine.set_linewidth(2)

        legend_handles = [mpatches.Patch(color=c, label=l) for c, l in zip(BAR_COLORS, BAR_LABELS)]
        fig.legend(handles=legend_handles, loc='upper center', ncol=3,
                   fontsize=9, frameon=True, bbox_to_anchor=(0.5, 1.0))
        fig.suptitle(f"Ablation at {iteration//1000}k steps",
                     fontsize=12, fontweight='bold', y=1.06)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        path = plots_dir / f"ablation_ckpt{iteration}.png"
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
