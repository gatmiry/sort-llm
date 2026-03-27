"""
Per-number intervention experiment for 100k checkpoints.
Fixes a target number (value) from the vocabulary, generates sequences
containing it, finds its sorted-output position, and intervenes there.
5 target numbers spread across vocab: 25, 75, 128, 180, 230.
Intensity values: [1.0, 2.0, 4.0, 6.0, 8.0, 10.0], ub=lb=60.
One worker per GPU, plots assembled on the fly.
"""
import json
import os
import subprocess
import sys
import time
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(SCRIPT_DIR, 'final_models')
OUTPUT_BASE = os.path.join(SCRIPT_DIR, 'outputs')

NUM_GPUS = 8
TARGET_NUMBERS = [25, 75, 128, 180, 230]
INTENSITY_VALUES = [1.0, 2.0, 4.0, 6.0, 8.0, 10.0]


def discover_checkpoints():
    pt_files = sorted(glob.glob(os.path.join(CKPT_DIR, '*.pt')))
    checkpoints = []
    for pt in pt_files:
        bn = os.path.basename(pt)
        if '.summary.' in bn:
            continue
        parts = bn.replace('.pt', '').split('__')
        config_str = parts[0]
        ckpt_type = parts[1] if len(parts) > 1 else 'final'
        tokens = config_str.split('_')
        dseed = iseed = None
        for t in tokens:
            if t.startswith('dseed'):
                dseed = t.replace('dseed', '')
            elif t.startswith('iseed'):
                iseed = t.replace('iseed', '')
        if ckpt_type.startswith('ckpt'):
            itr = int(ckpt_type.replace('ckpt', ''))
            ckpt_label = f'ckpt{itr}'
        else:
            itr = 100000
            ckpt_label = 'final'
        folder_name = f"plots_N256_B16_ds{dseed}_is{iseed}_{ckpt_label}"
        checkpoints.append({
            'path': pt, 'dseed': dseed, 'iseed': iseed,
            'itr': itr, 'ckpt_label': ckpt_label, 'folder_name': folder_name,
        })
    return checkpoints


def make_tasks(checkpoints):
    tasks = []
    for ckpt in checkpoints:
        tmp_dir = os.path.join(OUTPUT_BASE, 'tmp_results', ckpt['folder_name'], 'pernumber')
        for num in TARGET_NUMBERS:
            for layer in [0, 1]:
                out = os.path.join(tmp_dir, f'intensity_num{num}_layer{layer}.npz')
                tasks.append({
                    'ckpt_path': ckpt['path'],
                    'folder_name': ckpt['folder_name'],
                    'target_num': num,
                    'layer': layer,
                    'out': out,
                    'itr': ckpt['itr'],
                    'dseed': ckpt['dseed'],
                    'iseed': ckpt['iseed'],
                    'name': f"{ckpt['folder_name']}_num{num}_L{layer}",
                })
    return tasks


def is_ckpt_done(ckpt):
    tmp_dir = os.path.join(OUTPUT_BASE, 'tmp_results', ckpt['folder_name'], 'pernumber')
    for num in TARGET_NUMBERS:
        for layer in [0, 1]:
            if not os.path.exists(os.path.join(tmp_dir, f'intensity_num{num}_layer{layer}.npz')):
                return False
    return True


def assemble_plots_for_ckpt(ckpt):
    folder_name = ckpt['folder_name']
    tmp_dir = os.path.join(OUTPUT_BASE, 'tmp_results', folder_name, 'pernumber')
    plot_dir = os.path.join(OUTPUT_BASE, folder_name, 'pernumber')
    os.makedirs(plot_dir, exist_ok=True)

    tag = (f"N=256  block=16  lr=0.01  iters={ckpt['itr']}  "
           f"dseed={ckpt['dseed']}  iseed={ckpt['iseed']}")

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    markers = ['o', 's', '^', 'D', 'v']

    for layer in [0, 1]:
        fig, ax = plt.subplots(figsize=(5.5, 4))
        for i, num in enumerate(TARGET_NUMBERS):
            f = os.path.join(tmp_dir, f'intensity_num{num}_layer{layer}.npz')
            if not os.path.exists(f):
                continue
            d = np.load(f)
            ax.plot(d['intensities'], d['success_rates'],
                    marker=markers[i], linewidth=1.5, markersize=5,
                    color=colors[i], label=f'number {num}')

        ax.set_xlabel('Intervention Intensity', fontsize=10)
        ax.set_ylabel('Success Probability', fontsize=10)
        title = f'Per-Number Intervention (Layer {layer}, ub=60)\n{tag}'
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.legend(fontsize=8, loc='lower left')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(INTENSITY_VALUES)
        ax.set_ylim(0, 1.05)
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, f'pernumber_layer{layer}.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()
    return 2


def main():
    t_start = time.time()
    checkpoints = discover_checkpoints()
    print(f"Found {len(checkpoints)} checkpoints")

    all_tasks = make_tasks(checkpoints)
    todo = [t for t in all_tasks if not os.path.exists(t['out'])]
    cached = len(all_tasks) - len(todo)
    print(f"Total tasks: {len(all_tasks)}, cached: {cached}, to run: {len(todo)}")

    assembled = set()
    for ckpt in checkpoints:
        if is_ckpt_done(ckpt):
            assemble_plots_for_ckpt(ckpt)
            assembled.add(ckpt['folder_name'])
            print(f"  [PLOTS] {ckpt['folder_name']} (cached)", flush=True)

    if not todo:
        print("All done!")
        return

    gpu_tasks = {g: [] for g in range(NUM_GPUS)}
    ckpt_to_gpu = {}
    for i, ckpt in enumerate(checkpoints):
        ckpt_to_gpu[ckpt['folder_name']] = i % NUM_GPUS

    for t in todo:
        g = ckpt_to_gpu[t['folder_name']]
        gpu_tasks[g].append(t)

    for g in gpu_tasks:
        gpu_tasks[g].sort(key=lambda t: (t['ckpt_path'], t['target_num'], t['layer']))

    total_to_run = len(todo)
    print(f"\nDistributed {total_to_run} tasks across {NUM_GPUS} GPUs:")
    for g in range(NUM_GPUS):
        n = len(gpu_tasks[g])
        ckpts = len(set(t['ckpt_path'] for t in gpu_tasks[g])) if n else 0
        print(f"  GPU {g}: {n} tasks across {ckpts} checkpoints")

    task_dir = os.path.join(OUTPUT_BASE, 'task_files')
    os.makedirs(task_dir, exist_ok=True)
    log_dir = os.path.join(OUTPUT_BASE, 'pernumber_logs')
    os.makedirs(log_dir, exist_ok=True)

    procs = {}
    for g in range(NUM_GPUS):
        if not gpu_tasks[g]:
            continue
        tf = os.path.join(task_dir, f'pernum_gpu{g}.json')
        with open(tf, 'w') as f:
            json.dump(gpu_tasks[g], f)
        log_file = open(os.path.join(log_dir, f'gpu{g}.log'), 'w')
        proc = subprocess.Popen(
            [sys.executable, os.path.join(SCRIPT_DIR, 'pernumber_worker.py'),
             '--tasks-file', tf, '--gpu', str(g)],
            stdout=log_file, stderr=subprocess.STDOUT, cwd=SCRIPT_DIR)
        procs[g] = proc

    print(f"\nLaunched {len(procs)} workers. Monitoring...\n", flush=True)

    last_print = 0
    while any(p.poll() is None for p in procs.values()):
        time.sleep(5)
        for ckpt in checkpoints:
            fn = ckpt['folder_name']
            if fn not in assembled and is_ckpt_done(ckpt):
                assemble_plots_for_ckpt(ckpt)
                assembled.add(fn)
                elapsed = time.time() - t_start
                print(f"  [PLOTS] {fn}: 2 plots ({elapsed:.0f}s)", flush=True)

        done_now = sum(1 for t in all_tasks if os.path.exists(t['out']))
        elapsed = time.time() - t_start
        if done_now >= last_print + 10:
            last_print = done_now
            rate = done_now / elapsed if elapsed > 0 else 0
            eta = (len(all_tasks) - done_now) / rate if rate > 0 else 0
            print(f"  [PROGRESS] {done_now}/{len(all_tasks)} tasks, "
                  f"{len(assembled)}/{len(checkpoints)} ckpts plotted "
                  f"({elapsed:.0f}s, ETA ~{eta:.0f}s)", flush=True)

    for ckpt in checkpoints:
        fn = ckpt['folder_name']
        if fn not in assembled and is_ckpt_done(ckpt):
            assemble_plots_for_ckpt(ckpt)
            assembled.add(fn)
            print(f"  [PLOTS] {fn}: 2 plots (final)", flush=True)

    for g, proc in procs.items():
        if proc.returncode != 0:
            print(f"  [WARN] GPU {g} exited with code {proc.returncode}", flush=True)

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"ALL DONE — {len(assembled)}/{len(checkpoints)} checkpoints plotted")
    print(f"Elapsed: {elapsed:.0f}s ({elapsed/60:.1f}m)")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
