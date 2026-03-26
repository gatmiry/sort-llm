"""
Launch separator-attention and random-target intervention experiments across 8 GPUs.
Each GPU runs N trials, collecting per-number success data.
Assembles two plots:
  1. Success probability per number when intervening at separator-attending positions
  2. Success probability per number when intervening with random target
"""
import os
import subprocess
import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_BASE = os.path.join(SCRIPT_DIR, 'outputs')
PLOT_DIR = os.path.join(OUTPUT_BASE,
    'plots_V256_B16_LR3e-2_MI200000_E64_H1_L2_ds1337_is1337_ckpt200000')
TMP_DIR = os.path.join(OUTPUT_BASE, 'tmp_results', 'separator_random')
CKPT = os.path.join(SCRIPT_DIR,
    'sortgpt_k16_methfixed_mlp1_L2_N256_E64_pos0_fln1_wd0p0_lr0p03_dseed1337_iseed1337__final.pt')

NUM_GPUS = 8
TRIALS_PER_GPU = 1000
INTENSITIES = [2.0, 6.0, 10.0]
TAG = 'V=256  B=16  lr=0.03  iters=200000  dseed=1337  iseed=1337'


def launch_workers():
    os.makedirs(TMP_DIR, exist_ok=True)
    log_dir = os.path.join(OUTPUT_BASE, 'separator_logs')
    os.makedirs(log_dir, exist_ok=True)

    procs = []
    for g in range(NUM_GPUS):
        out = os.path.join(TMP_DIR, f'gpu{g}.npz')
        if os.path.exists(out):
            continue
        lf = open(os.path.join(log_dir, f'gpu{g}.log'), 'w')
        proc = subprocess.Popen(
            [sys.executable, os.path.join(SCRIPT_DIR, 'separator_intervention_worker.py'),
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
        print(f"  [{time.time()-t0:.0f}s] {done}/{NUM_GPUS} GPUs done", flush=True)

    for proc, lf, g in procs:
        lf.close()
        if proc.returncode != 0:
            print(f"  WARN: GPU {g} exited with code {proc.returncode}", flush=True)

    print(f"All workers done in {time.time()-t0:.0f}s", flush=True)


def load_and_combine():
    all_sep, all_rand = [], []
    for g in range(NUM_GPUS):
        f = os.path.join(TMP_DIR, f'gpu{g}.npz')
        if not os.path.exists(f):
            continue
        d = np.load(f)
        if len(d['sep_data']) > 0:
            all_sep.append(d['sep_data'])
        if len(d['rand_data']) > 0:
            all_rand.append(d['rand_data'])

    sep = np.concatenate(all_sep) if all_sep else np.empty((0, 3), dtype=np.int32)
    rand = np.concatenate(all_rand) if all_rand else np.empty((0, 3), dtype=np.int32)
    print(f"Combined: sep={len(sep)} records, rand={len(rand)} records")
    return sep, rand


def plot_per_number(data, title_prefix, filename, tag):
    """Plot success probability per number for each intensity."""
    os.makedirs(PLOT_DIR, exist_ok=True)

    colors = {2.0: '#1f77b4', 6.0: '#ff7f0e', 10.0: '#d62728'}
    fig, axes = plt.subplots(2, 1, figsize=(14, 8),
                             gridspec_kw={'height_ratios': [3, 1]})

    ax = axes[0]
    for intens in INTENSITIES:
        mask = data[:, 1] == intens
        subset = data[mask]
        if len(subset) == 0:
            continue

        xs, ys = [], []
        for n_val in range(256):
            nm = subset[:, 0] == n_val
            count = nm.sum()
            if count >= 10:
                xs.append(n_val)
                ys.append(subset[nm, 2].mean())

        ax.plot(xs, ys, color=colors.get(intens, '#333'),
                linewidth=0.8, alpha=0.6, label=f'raw int={intens}')

        if len(xs) >= 11:
            raw_arr = np.full(256, np.nan)
            for x, y in zip(xs, ys):
                raw_arr[x] = y
            win = 11
            padded = np.nan_to_num(raw_arr, nan=0.5)
            smoothed = np.convolve(padded, np.ones(win) / win, mode='same')
            valid = ~np.isnan(raw_arr)
            ax.plot(np.arange(256)[valid], smoothed[valid],
                    color=colors.get(intens, '#333'), linewidth=2.5,
                    linestyle='--', label=f'smoothed int={intens}')

    ax.set_ylabel('Success Probability', fontsize=12)
    ax.set_title(f'{title_prefix} (Layer 0)\n{tag}', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, ncol=2, loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.1)
    ax.set_xlim(0, 255)

    ax2 = axes[1]
    max_intens = max(INTENSITIES)
    mask_hi = data[:, 1] == max_intens
    counts = np.array([(mask_hi & (data[:, 0] == n)).sum() for n in range(256)])
    ax2.bar(range(256), counts, width=1, color='#666', alpha=0.5)
    ax2.set_xlabel('Number in Vocabulary', fontsize=12)
    ax2.set_ylabel('Sample Count', fontsize=10)
    ax2.set_xlim(0, 255)
    ax2.grid(True, alpha=0.2)

    fig.tight_layout()
    out_path = os.path.join(PLOT_DIR, filename)
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Plot saved: {out_path}")


def main():
    t0 = time.time()
    print("=" * 60)
    print("SEPARATOR & RANDOM INTERVENTION EXPERIMENT (Layer 0)")
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

    sep, rand = load_and_combine()

    if len(sep) > 0:
        plot_per_number(sep,
            'Intervention Success when Attending to Separator',
            'intervention_pernumber_separator_layer0.png', TAG)
    else:
        print("WARNING: No separator-attending data collected!")

    if len(rand) > 0:
        plot_per_number(rand,
            'Intervention Success with Random Target',
            'intervention_pernumber_random_layer0.png', TAG)

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f}m)")


if __name__ == '__main__':
    main()
