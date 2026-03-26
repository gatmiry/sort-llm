"""
Launch hijack intervention experiment across 8 GPUs.
For each sorted position, boost a random unsorted number's attention (intensity=10),
record what model predicts. Then produce two heatmaps:
  1. Breaking rate: P(predicted != correct_next) per (current_num, boosted_num) pair
  2. Hijack rate: P(predicted == boosted_num) per (current_num, boosted_num) pair
"""
import os
import subprocess
import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_BASE = os.path.join(SCRIPT_DIR, 'outputs')
PLOT_DIR = os.path.join(OUTPUT_BASE,
    'plots_V256_B16_LR3e-2_MI200000_E64_H1_L2_ds1337_is1337_ckpt200000')
TMP_DIR = os.path.join(OUTPUT_BASE, 'tmp_results', 'hijack')
CKPT = os.path.join(SCRIPT_DIR,
    'sortgpt_k16_methfixed_mlp1_L2_N256_E64_pos0_fln1_wd0p0_lr0p03_dseed1337_iseed1337__final.pt')

NUM_GPUS = 8
TRIALS_PER_GPU = 10000
TAG = 'V=256  B=16  lr=0.03  200k iters  intensity=10'
BIN_SIZE = 8
N_BINS = 256 // BIN_SIZE


def launch_workers():
    os.makedirs(TMP_DIR, exist_ok=True)
    log_dir = os.path.join(OUTPUT_BASE, 'hijack_logs')
    os.makedirs(log_dir, exist_ok=True)

    procs = []
    for g in range(NUM_GPUS):
        out = os.path.join(TMP_DIR, f'gpu{g}.npz')
        if os.path.exists(out):
            continue
        lf = open(os.path.join(log_dir, f'gpu{g}.log'), 'w')
        proc = subprocess.Popen(
            [sys.executable, os.path.join(SCRIPT_DIR, 'hijack_intervention_worker.py'),
             '--ckpt', CKPT, '--gpu', str(g),
             '--trials', str(TRIALS_PER_GPU), '--out', out],
            stdout=lf, stderr=subprocess.STDOUT, cwd=SCRIPT_DIR)
        procs.append((proc, lf, g))
    return procs


def wait_for_workers(procs):
    t0 = time.time()
    while any(p.poll() is None for p, _, _ in procs):
        time.sleep(10)
        done = sum(1 for g in range(NUM_GPUS)
                   if os.path.exists(os.path.join(TMP_DIR, f'gpu{g}.npz')))
        elapsed = time.time() - t0
        print(f"  [{elapsed:.0f}s] {done}/{NUM_GPUS} GPUs done", flush=True)

    for proc, lf, g in procs:
        lf.close()
        if proc.returncode != 0:
            print(f"  WARN: GPU {g} exited with code {proc.returncode}", flush=True)
    print(f"All workers done in {time.time()-t0:.0f}s", flush=True)


def load_and_combine():
    all_data = []
    for g in range(NUM_GPUS):
        f = os.path.join(TMP_DIR, f'gpu{g}.npz')
        if not os.path.exists(f):
            continue
        d = np.load(f)
        if len(d['data']) > 0:
            all_data.append(d['data'])
    data = np.concatenate(all_data) if all_data else np.empty((0, 4), dtype=np.int32)
    print(f"Combined: {len(data)} records")
    return data


def make_heatmaps(data):
    os.makedirs(PLOT_DIR, exist_ok=True)

    current = data[:, 0]
    boosted = data[:, 1]
    predicted = data[:, 2]
    correct = data[:, 3]

    broken = (predicted != correct).astype(np.float64)
    hijacked = (predicted == boosted).astype(np.float64)

    cur_bin = np.clip(current // BIN_SIZE, 0, N_BINS - 1)
    bst_bin = np.clip(boosted // BIN_SIZE, 0, N_BINS - 1)

    break_map = np.full((N_BINS, N_BINS), np.nan)
    hijack_map = np.full((N_BINS, N_BINS), np.nan)
    count_map = np.zeros((N_BINS, N_BINS), dtype=int)

    for cb in range(N_BINS):
        for bb in range(N_BINS):
            mask = (cur_bin == cb) & (bst_bin == bb)
            n = mask.sum()
            count_map[cb, bb] = n
            if n >= 5:
                break_map[cb, bb] = broken[mask].mean()
                hijack_map[cb, bb] = hijacked[mask].mean()

    tick_labels = [f'{i * BIN_SIZE}' for i in range(0, N_BINS, 4)]
    tick_positions = list(range(0, N_BINS, 4))

    # Breaking rate heatmap
    fig, ax = plt.subplots(figsize=(10, 8.5))
    im = ax.imshow(break_map, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1,
                   interpolation='nearest', origin='lower')
    ax.set_xlabel('Intervened-toward Number (binned)', fontsize=12)
    ax.set_ylabel('Current Number (binned)', fontsize=12)
    ax.set_title(f'Breaking Rate: P(pred ≠ correct) per (current, target) pair\n{TAG}',
                 fontsize=12, fontweight='bold')
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=8)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels, fontsize=8)
    plt.colorbar(im, ax=ax, label='Breaking Rate', shrink=0.85)
    fig.tight_layout()
    out1 = os.path.join(PLOT_DIR, 'hijack_breaking_rate_heatmap_layer0.png')
    fig.savefig(out1, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out1}")

    # Hijack rate heatmap
    fig, ax = plt.subplots(figsize=(10, 8.5))
    im = ax.imshow(hijack_map, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1,
                   interpolation='nearest', origin='lower')
    ax.set_xlabel('Intervened-toward Number (binned)', fontsize=12)
    ax.set_ylabel('Current Number (binned)', fontsize=12)
    ax.set_title(f'Hijack Rate: P(pred == intervened target) per (current, target) pair\n{TAG}',
                 fontsize=12, fontweight='bold')
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=8)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels, fontsize=8)
    plt.colorbar(im, ax=ax, label='Hijack Rate', shrink=0.85)
    fig.tight_layout()
    out2 = os.path.join(PLOT_DIR, 'hijack_hijack_rate_heatmap_layer0.png')
    fig.savefig(out2, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out2}")

    # Sample count heatmap
    fig, ax = plt.subplots(figsize=(10, 8.5))
    im = ax.imshow(count_map, aspect='auto', cmap='viridis',
                   interpolation='nearest', origin='lower')
    ax.set_xlabel('Intervened-toward Number (binned)', fontsize=12)
    ax.set_ylabel('Current Number (binned)', fontsize=12)
    ax.set_title(f'Sample Count per (current, target) bin\n{TAG}',
                 fontsize=11, fontweight='bold')
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=8)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels, fontsize=8)
    plt.colorbar(im, ax=ax, label='Count', shrink=0.85)
    fig.tight_layout()
    out3 = os.path.join(PLOT_DIR, 'hijack_sample_count_heatmap_layer0.png')
    fig.savefig(out3, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out3}")

    # Summary stats
    valid = ~np.isnan(break_map)
    print(f"\nSummary:")
    print(f"  Total records: {len(data)}")
    print(f"  Bins with >=5 samples: {valid.sum()}/{N_BINS*N_BINS}")
    print(f"  Overall breaking rate: {broken.mean():.4f}")
    print(f"  Overall hijack rate: {hijacked.mean():.4f}")
    print(f"  Mean breaking rate (binned): {np.nanmean(break_map):.4f}")
    print(f"  Mean hijack rate (binned): {np.nanmean(hijack_map):.4f}")


def main():
    t0 = time.time()
    print("=" * 60)
    print("HIJACK INTERVENTION EXPERIMENT (Layer 0, intensity=10)")
    print("=" * 60)

    cached = sum(1 for g in range(NUM_GPUS)
                 if os.path.exists(os.path.join(TMP_DIR, f'gpu{g}.npz')))
    print(f"Cached GPUs: {cached}/{NUM_GPUS}")

    procs = launch_workers()
    if procs:
        print(f"Launched {len(procs)} workers")
        wait_for_workers(procs)
    else:
        print("All workers already cached")

    data = load_and_combine()
    if len(data) > 0:
        make_heatmaps(data)

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f}m)")


if __name__ == '__main__':
    main()
