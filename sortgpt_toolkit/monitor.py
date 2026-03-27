#!/usr/bin/env python3
"""
Live monitor for multi-seed SortGPT training.

Polls for new checkpoints. When ALL alive seeds reach a checkpoint:
  - Evaluates ablation (full / no-attn-1 / no-attn-2)
  - Evaluates length generalization (26 lengths, 4–256)
  - Generates bar plots and length gen plots
  - Auto-kills diverged seeds (0 accuracy after 100k)
  - Writes color-coded status.txt:
      GREEN = no_attn2 > 0.95 (single-layer solution)
      RED   = no_attn2 < 0.95 (two-layer solution)

Usage:
    CUDA_VISIBLE_DEVICES=X python monitor.py \\
        --run-dir DIR --seeds 1501 1502 ... --init-stds 0.02 0.02 ... \\
        --checkpoint-every 50000 --max-iters 1000000
"""

import argparse
import gc
import os
import signal
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch

from model import DEVICE, float_token, load_model_from_checkpoint
from evaluate import (
    evaluate_ablation, evaluate_length_generalization, EVAL_LENGTHS
)

warnings.filterwarnings("ignore", category=DeprecationWarning)

POLL_INTERVAL_SEC = 30
VOCAB_N = 256
BLOCK_SIZE = 16
EXTENDED_MAX_SEQ_LEN = 2 * VOCAB_N + 1


# ── Status file ──────────────────────────────────────────────────────────────

def write_status(run_dir, seeds, init_stds, seed_results, killed_seeds):
    lines = []
    lines.append("=== SortGPT Training Monitor ===")
    lines.append(f"Updated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append(f"{'Seed':>6} {'Std':>8} {'Status':>10} {'LastCkpt':>10} "
                 f"{'Full':>8} {'NoAttn1':>8} {'NoAttn2':>8} {'Color':>8}")
    lines.append("-" * 78)

    for seed, std in zip(seeds, init_stds):
        if seed in killed_seeds:
            status = "KILLED"
            s = killed_seeds[seed]
            last_ckpt = f"{s.get('iter', 0)//1000}k"
            full, no1, no2 = s.get("full_tok", 0), s.get("no_attn1_tok", 0), s.get("no_attn2_tok", 0)
        elif seed in seed_results and seed_results[seed]:
            latest = seed_results[seed][-1]
            status = "RUNNING"
            last_ckpt = f"{latest['iter']//1000}k"
            full, no1, no2 = latest["full_tok"], latest["no_attn1_tok"], latest["no_attn2_tok"]
        else:
            pid_file = run_dir / "pids" / f"seed_{seed}.pid"
            if pid_file.exists():
                try:
                    pid = int(pid_file.read_text().strip())
                    os.kill(pid, 0)
                    status = "TRAINING"
                except (ProcessLookupError, ValueError):
                    status = "DONE/DEAD"
            else:
                status = "UNKNOWN"
            last_ckpt = "-"
            full = no1 = no2 = 0.0

        if seed in killed_seeds:
            color = "DIVERGED"
        elif full < 0.01:
            color = "DIVERGED"
        elif no2 > 0.95:
            color = "GREEN"
        else:
            color = "RED"

        lines.append(f"{seed:>6} {std:>8.4f} {status:>10} {last_ckpt:>10} "
                     f"{full:>8.4f} {no1:>8.4f} {no2:>8.4f} {color:>8}")

    lines.append("")
    lines.append("GREEN = no_attn2 > 0.95 (single-layer solution)")
    lines.append("RED   = no_attn2 < 0.95 (two-layer / not yet converged)")
    lines.append("DIVERGED = killed due to 0 accuracy")
    lines.append("")
    lines.append(f"To stop seed X:  kill $(cat {run_dir}/pids/seed_X.pid)")
    lines.append(f"To stop all:     kill $(cat {run_dir}/pids/*.pid)")

    (run_dir / "status.txt").write_text("\n".join(lines) + "\n")


# ── Plots ────────────────────────────────────────────────────────────────────

def make_ablation_plot(seed_scores, iteration, plots_dir):
    seeds = sorted(seed_scores.keys())
    n = len(seeds)
    if n == 0:
        return

    BAR_COLORS = ['#4C72B0', '#DD8452', '#55A868']
    BAR_LABELS = ['Full model', 'Attn-1 removed', 'Attn-2 removed']
    CONDITIONS = ['full_tok', 'no_attn1_tok', 'no_attn2_tok']

    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 5), squeeze=False)
    for col, seed in enumerate(seeds):
        ax = axes[0][col]
        scores = seed_scores[seed]
        no2 = scores.get("no_attn2_tok", 0)
        border = "green" if no2 > 0.95 else "red"

        for i, (cond, color, lbl) in enumerate(zip(CONDITIONS, BAR_COLORS, BAR_LABELS)):
            val = scores.get(cond, 0)
            ax.bar(i, val, width=0.55, color=color, edgecolor='white', linewidth=0.8, label=lbl)
            ax.text(i, val + 0.012, f"{val:.3f}", ha='center', va='bottom', fontsize=9, fontweight='bold')

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
    fig.suptitle(f"Attention Ablation — {iteration//1000}k iterations\n"
                 f"(k={BLOCK_SIZE}, N={VOCAB_N})",
                 fontsize=12, y=1.06, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = plots_dir / f"ablation_ckpt{iteration}.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


def make_lengthgen_plot(seed_lengthgen, iteration, plots_dir, seed_results, killed_seeds):
    seeds = sorted(seed_lengthgen.keys())
    if not seeds:
        return

    fig, ax = plt.subplots(figsize=(12, 7))
    for seed in seeds:
        accs = seed_lengthgen[seed]
        if seed in killed_seeds:
            color, ls, lbl = "gray", ":", f"seed={seed} (diverged)"
        elif seed in seed_results and seed_results[seed]:
            no2 = seed_results[seed][-1].get("no_attn2_tok", 0)
            color = "green" if no2 > 0.95 else "red"
            ls = "-"
            lbl = f"seed={seed} (no2={no2:.2f})"
        else:
            color, ls, lbl = "blue", "-", f"seed={seed}"
        ax.plot(EVAL_LENGTHS, accs, ls, color=color, linewidth=2, markersize=4,
                marker="o", label=lbl, alpha=0.85)

    ax.axvline(x=BLOCK_SIZE, color="gray", linestyle="--", linewidth=1, alpha=0.7,
               label=f"Training length (k={BLOCK_SIZE})")
    ax.set_xlabel("Sequence Length", fontsize=13)
    ax.set_ylabel("Token Accuracy", fontsize=13)
    ax.set_title(f"Length Generalization — {iteration//1000}k iterations\n"
                 f"(k={BLOCK_SIZE}, N={VOCAB_N})",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(-0.02, 1.05)
    ax.set_xticks(EVAL_LENGTHS[::2])
    ax.set_xticklabels([str(x) for x in EVAL_LENGTHS[::2]], fontsize=8, rotation=45, ha="right")
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = plots_dir / f"lengthgen_ckpt{iteration}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ── Main loop ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--init-stds", type=float, nargs="+", required=True)
    parser.add_argument("--checkpoint-every", type=int, default=50_000)
    parser.add_argument("--max-iters", type=int, default=1_000_000)
    args = parser.parse_args()

    from pathlib import Path
    run_dir = Path(args.run_dir)
    ckpt_dir = run_dir / "checkpoints"
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    seeds = args.seeds
    init_stds = args.init_stds
    assert len(seeds) == len(init_stds), "seeds and init_stds must have same length"
    std_by_seed = dict(zip(seeds, init_stds))

    checkpoint_iters = list(range(args.checkpoint_every, args.max_iters + 1, args.checkpoint_every))

    gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    print(f"=== Monitor on GPU {gpu_id} ===")
    print(f"    Seeds: {seeds}")
    print(f"    Stds:  {init_stds}")
    print(f"    Checkpoints: {[f'{x//1000}k' for x in checkpoint_iters]}")

    evaluated = {s: set() for s in seeds}
    seed_results = {s: [] for s in seeds}
    killed_seeds = {}

    write_status(run_dir, seeds, init_stds, seed_results, killed_seeds)

    while True:
        any_new = False

        for iteration in checkpoint_iters:
            alive_seeds = [s for s in seeds if s not in killed_seeds]

            # Evaluate any unevaluated seeds at this checkpoint
            for seed in alive_seeds:
                if iteration in evaluated[seed]:
                    continue
                tag = f"std{float_token(std_by_seed[seed])}_iseed{seed}"
                ckpt_path = ckpt_dir / f"{tag}__ckpt{iteration}.pt"
                if not ckpt_path.exists():
                    continue

                time.sleep(2)
                print(f"\n  Evaluating seed={seed} at {iteration//1000}k...")
                try:
                    model = load_model_from_checkpoint(ckpt_path,
                                                       extended_max_seq_len=EXTENDED_MAX_SEQ_LEN)
                except Exception as e:
                    print(f"    ERROR loading: {e}")
                    continue

                scores = evaluate_ablation(model, BLOCK_SIZE, VOCAB_N)
                scores["iter"] = iteration
                scores["seed"] = seed

                print(f"    full={scores['full_tok']:.4f}  no_attn1={scores['no_attn1_tok']:.4f}  "
                      f"no_attn2={scores['no_attn2_tok']:.4f}  samp_acc={scores['full_samp']:.4f}")

                print(f"    Evaluating length generalization...")
                lg_accs = evaluate_length_generalization(model, VOCAB_N)
                scores["lengthgen"] = lg_accs
                L16i = EVAL_LENGTHS.index(16)
                L64i = EVAL_LENGTHS.index(64)
                L256i = EVAL_LENGTHS.index(256)
                print(f"    L16={lg_accs[L16i]:.4f}  L64={lg_accs[L64i]:.4f}  L256={lg_accs[L256i]:.4f}")

                seed_results[seed].append(scores)
                evaluated[seed].add(iteration)
                any_new = True

                # Auto-kill diverged seeds
                if scores["full_samp"] < 0.001 and iteration >= 100_000:
                    print(f"    *** DIVERGED — killing seed {seed} ***")
                    pid_file = run_dir / "pids" / f"seed_{seed}.pid"
                    if pid_file.exists():
                        try:
                            pid = int(pid_file.read_text().strip())
                            os.kill(pid, signal.SIGTERM)
                            print(f"    Sent SIGTERM to PID {pid}")
                        except (ProcessLookupError, ValueError):
                            print(f"    Process already dead")
                    killed_seeds[seed] = scores

                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

            # Generate plots only when ALL alive seeds have this checkpoint
            alive_seeds = [s for s in seeds if s not in killed_seeds]
            if alive_seeds and all(iteration in evaluated[s] for s in alive_seeds):
                all_scores = {}
                for s in alive_seeds:
                    for r in seed_results[s]:
                        if r["iter"] == iteration:
                            all_scores[s] = r
                            break

                if all_scores:
                    print(f"\n  All {len(alive_seeds)} alive seeds at {iteration//1000}k — plotting...")
                    path = make_ablation_plot(all_scores, iteration, plots_dir)
                    if path:
                        print(f"  Saved: {path}")
                    seed_lg = {s: sc["lengthgen"] for s, sc in all_scores.items() if "lengthgen" in sc}
                    if seed_lg:
                        lg_path = make_lengthgen_plot(seed_lg, iteration, plots_dir,
                                                      seed_results, killed_seeds)
                        if lg_path:
                            print(f"  Saved: {lg_path}")

        write_status(run_dir, seeds, init_stds, seed_results, killed_seeds)

        alive_seeds = [s for s in seeds if s not in killed_seeds]
        all_done = all(len(evaluated[s]) >= len(checkpoint_iters) for s in alive_seeds) if alive_seeds else True

        if all_done:
            print("\n=== All checkpoints evaluated. Done. ===")
            break

        # Check if all training processes are dead
        all_dead = True
        for seed in alive_seeds:
            pid_file = run_dir / "pids" / f"seed_{seed}.pid"
            if pid_file.exists():
                try:
                    pid = int(pid_file.read_text().strip())
                    os.kill(pid, 0)
                    all_dead = False
                except (ProcessLookupError, ValueError):
                    pass

        if all_dead and not any_new:
            time.sleep(5)
            has_unevaluated = False
            for seed in alive_seeds:
                for iteration in checkpoint_iters:
                    if iteration not in evaluated[seed]:
                        tag = f"std{float_token(std_by_seed[seed])}_iseed{seed}"
                        if (ckpt_dir / f"{tag}__ckpt{iteration}.pt").exists():
                            has_unevaluated = True
                            break
            if not has_unevaluated:
                print("\n=== All training finished. Exiting. ===")
                break

        remaining = sum(len(checkpoint_iters) - len(evaluated[s]) for s in alive_seeds)
        print(f"  [{remaining} evals remaining, polling every {POLL_INTERVAL_SEC}s...]")
        time.sleep(POLL_INTERVAL_SEC)


if __name__ == "__main__":
    main()
