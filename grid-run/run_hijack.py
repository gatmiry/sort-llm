"""
Run hijack intervention for all grid-run checkpoints.
For each (V, B, LR, MI) pair, runs hijack on both LN0 and LN1 final checkpoints.
Produces 3 heatmaps per model (breaking rate, hijack rate, sample count),
saved in the corresponding plots_xxx folder with LN0/LN1 suffix.
Distributed across 8 GPUs with incremental plot assembly.
"""
import itertools
import json
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
NUM_GPUS = 8
BIN_SIZE = 8


def find_final_ckpt(run_dir, max_iters):
    if not os.path.isdir(run_dir):
        return None
    target = f"_itr{max_iters}.pt"
    for f in os.listdir(run_dir):
        if f.endswith(target):
            return os.path.join(run_dir, f)
    return None


def discover_tasks():
    VOCAB_SIZES = [64, 128, 256, 512, 8192]
    BLOCK_SIZES = [16, 32]
    MAX_ITERS = [10000, 20000, 40000, 60000]
    LEARNING_RATES = ['1e-04', '1e-03', '1e-02']

    tasks = []
    for vs, bs, lr, mi in itertools.product(VOCAB_SIZES, BLOCK_SIZES, LEARNING_RATES, MAX_ITERS):
        plot_folder = f"plots_V{vs}_B{bs}_LR{lr}_MI{mi}_E64_H1_L2_ckpt{mi}"
        plot_dir = os.path.join(OUTPUT_BASE, plot_folder)
        if not os.path.isdir(plot_dir):
            continue

        for ln in [0, 1]:
            run_dir = os.path.join(OUTPUT_BASE, f"V{vs}_B{bs}_LR{lr}_MI{mi}_LN{ln}_E64_H1_L2")
            ckpt = find_final_ckpt(run_dir, mi)
            if not ckpt:
                continue

            tmp_dir = os.path.join(OUTPUT_BASE, 'tmp_hijack')
            out_file = os.path.join(tmp_dir, f"V{vs}_B{bs}_LR{lr}_MI{mi}_LN{ln}_hijack.npz")

            tasks.append({
                'ckpt_path': ckpt,
                'out': out_file,
                'name': f"V{vs}_B{bs}_LR{lr}_MI{mi}_LN{ln}",
                'vocab_size': vs, 'block_size': bs, 'lr': lr, 'max_iters': mi,
                'ln': ln,
                'plot_folder': plot_folder,
                'trials': 2000,
            })
    return tasks


def make_heatmaps(data, vocab_size, plot_dir, ln_label, tag):
    n_bins = vocab_size // BIN_SIZE
    if n_bins < 2:
        n_bins = vocab_size
        bin_size_local = 1
    else:
        bin_size_local = BIN_SIZE

    current = data[:, 0]; boosted = data[:, 1]
    predicted = data[:, 2]; correct = data[:, 3]
    broken = (predicted != correct).astype(np.float64)
    hijacked = (predicted == boosted).astype(np.float64)
    cur_bin = np.clip(current // bin_size_local, 0, n_bins - 1)
    bst_bin = np.clip(boosted // bin_size_local, 0, n_bins - 1)

    break_map = np.full((n_bins, n_bins), np.nan)
    hijack_map = np.full((n_bins, n_bins), np.nan)
    count_map = np.zeros((n_bins, n_bins), dtype=int)
    for cb in range(n_bins):
        for bb in range(n_bins):
            mask = (cur_bin == cb) & (bst_bin == bb)
            n = mask.sum()
            count_map[cb, bb] = n
            if n >= 5:
                break_map[cb, bb] = broken[mask].mean()
                hijack_map[cb, bb] = hijacked[mask].mean()

    tick_step = max(1, n_bins // 8)
    tick_labels = [f'{i * bin_size_local}' for i in range(0, n_bins, tick_step)]
    tick_positions = list(range(0, n_bins, tick_step))

    for arr, cmap, label, suffix in [
        (break_map, 'YlOrRd', 'Breaking Rate', 'breaking_rate'),
        (hijack_map, 'YlOrRd', 'Hijack Rate', 'hijack_rate'),
    ]:
        fig, ax = plt.subplots(figsize=(10, 8.5))
        im = ax.imshow(arr, aspect='auto', cmap=cmap, vmin=0, vmax=1,
                       interpolation='nearest', origin='lower')
        ax.set_xlabel('Intervened-toward Number (binned)', fontsize=12)
        ax.set_ylabel('Current Number (binned)', fontsize=12)
        title_map = {'Breaking Rate': f'Breaking Rate: P(pred ≠ correct)',
                     'Hijack Rate': f'Hijack Rate: P(pred == intervened target)'}
        ax.set_title(f'{title_map[label]} — {ln_label}\n{tag}  intensity=10',
                     fontsize=12, fontweight='bold')
        ax.set_xticks(tick_positions); ax.set_xticklabels(tick_labels, fontsize=8)
        ax.set_yticks(tick_positions); ax.set_yticklabels(tick_labels, fontsize=8)
        plt.colorbar(im, ax=ax, label=label, shrink=0.85)
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, f'hijack_{suffix}_heatmap_layer0_{ln_label}.png'),
                    dpi=200, bbox_inches='tight')
        plt.close()

    fig, ax = plt.subplots(figsize=(10, 8.5))
    im = ax.imshow(count_map, aspect='auto', cmap='viridis',
                   interpolation='nearest', origin='lower')
    ax.set_xlabel('Intervened-toward Number (binned)', fontsize=12)
    ax.set_ylabel('Current Number (binned)', fontsize=12)
    ax.set_title(f'Sample Count per (current, target) bin — {ln_label}\n{tag}  intensity=10',
                 fontsize=11, fontweight='bold')
    ax.set_xticks(tick_positions); ax.set_xticklabels(tick_labels, fontsize=8)
    ax.set_yticks(tick_positions); ax.set_yticklabels(tick_labels, fontsize=8)
    plt.colorbar(im, ax=ax, label='Count', shrink=0.85)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, f'hijack_sample_count_heatmap_layer0_{ln_label}.png'),
                dpi=200, bbox_inches='tight')
    plt.close()


def try_assemble(task):
    f = task['out']
    if not os.path.exists(f):
        return False
    d = np.load(f)
    data = d['data']
    if len(data) == 0:
        return True
    vs = int(d['vocab_size'])
    plot_dir = os.path.join(OUTPUT_BASE, task['plot_folder'])
    ln_label = f"LN{task['ln']}"
    tag = f"V={task['vocab_size']}  B={task['block_size']}  lr={task['lr']}  iters={task['max_iters']}"
    make_heatmaps(data, vs, plot_dir, ln_label, tag)
    return True


def main():
    t_start = time.time()
    all_tasks = discover_tasks()
    cached = [t for t in all_tasks if os.path.exists(t['out'])]
    to_run = [t for t in all_tasks if not os.path.exists(t['out'])]
    print(f"Total tasks: {len(all_tasks)}, cached: {len(cached)}, to run: {len(to_run)}")

    assembled = set()
    for t in cached:
        try:
            try_assemble(t)
            assembled.add(t['name'])
        except:
            pass

    if not to_run:
        print("All cached. Assembling plots only...")
        for t in all_tasks:
            if t['name'] not in assembled:
                try_assemble(t)
                assembled.add(t['name'])
        print(f"Done! {len(assembled)} plots assembled")
        return

    gpu_tasks = {g: [] for g in range(NUM_GPUS)}
    for i, t in enumerate(to_run):
        gpu_tasks[i % NUM_GPUS].append(t)

    for g in gpu_tasks:
        gpu_tasks[g].sort(key=lambda t: t['ckpt_path'])

    print(f"\nDistributed {len(to_run)} tasks across {NUM_GPUS} GPUs:")
    for g in range(NUM_GPUS):
        print(f"  GPU {g}: {len(gpu_tasks[g])} tasks")

    task_dir = os.path.join(OUTPUT_BASE, 'hijack_task_files')
    log_dir = os.path.join(OUTPUT_BASE, 'hijack_logs')
    os.makedirs(task_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    procs = {}
    for g in range(NUM_GPUS):
        if not gpu_tasks[g]:
            continue
        tf = os.path.join(task_dir, f'gpu{g}.json')
        with open(tf, 'w') as f:
            json.dump(gpu_tasks[g], f)
        lf = open(os.path.join(log_dir, f'gpu{g}.log'), 'w')
        proc = subprocess.Popen(
            [sys.executable, os.path.join(SCRIPT_DIR, 'hijack_worker.py'),
             '--tasks-file', tf, '--gpu', str(g)],
            stdout=lf, stderr=subprocess.STDOUT, cwd=SCRIPT_DIR)
        procs[g] = (proc, lf)

    print(f"\nLaunched {len(procs)} workers. Monitoring...\n", flush=True)

    last_print = 0
    while any(p.poll() is None for p, _ in procs.values()):
        time.sleep(10)
        for t in all_tasks:
            if t['name'] not in assembled:
                try:
                    if try_assemble(t):
                        assembled.add(t['name'])
                        elapsed = time.time() - t_start
                        print(f"  [PLOTS] {t['name']}: 3 heatmaps ({elapsed:.0f}s)", flush=True)
                except:
                    pass

        done_now = sum(1 for t in all_tasks if os.path.exists(t['out']))
        elapsed = time.time() - t_start
        if done_now >= last_print + 10:
            last_print = done_now
            rate = done_now / elapsed if elapsed > 0 else 0
            eta = (len(all_tasks) - done_now) / rate if rate > 0 else 0
            print(f"  [PROGRESS] {done_now}/{len(all_tasks)} tasks, "
                  f"{len(assembled)} plotted ({elapsed:.0f}s, ETA ~{eta:.0f}s)", flush=True)

    for g, (proc, lf) in procs.items():
        lf.close()
        if proc.returncode != 0:
            print(f"  [WARN] GPU {g} exited with code {proc.returncode}", flush=True)

    for t in all_tasks:
        if t['name'] not in assembled:
            try:
                try_assemble(t)
                assembled.add(t['name'])
                print(f"  [PLOTS] {t['name']}: 3 heatmaps (final)", flush=True)
            except:
                pass

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"ALL DONE — {len(assembled)}/{len(all_tasks)} models plotted")
    print(f"Elapsed: {elapsed:.0f}s ({elapsed/60:.1f}m)")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
