"""
Parallel launcher for 100k-checkpoint analysis.
Assigns checkpoints to 8 GPUs (one persistent worker per GPU).
Assembles plots incrementally as each checkpoint completes.
"""
import json
import os
import subprocess
import sys
import time
import glob
import threading
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(SCRIPT_DIR, 'final_models')
OUTPUT_BASE = os.path.join(SCRIPT_DIR, 'outputs')

NUM_GPUS = 8
UB_VALUES = [5, 10, 15, 20, 30, 50, 60]


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
            'path': pt,
            'dseed': dseed,
            'iseed': iseed,
            'itr': itr,
            'ckpt_label': ckpt_label,
            'folder_name': folder_name,
        })
    return checkpoints


def make_tasks_for_checkpoint(ckpt):
    """Generate all analysis tasks for a single checkpoint."""
    tmp_dir = os.path.join(OUTPUT_BASE, 'tmp_results', ckpt['folder_name'])
    tasks = []

    tasks.append({
        'ckpt_path': ckpt['path'], 'type': 'baseline', 'layer': 0,
        'out': os.path.join(tmp_dir, 'baseline.npz'),
        'name': f"{ckpt['folder_name']}_baseline",
        'itr': ckpt['itr'],
    })

    for layer in [0, 1]:
        tasks.append({
            'ckpt_path': ckpt['path'], 'type': 'ablation', 'layer': layer,
            'out': os.path.join(tmp_dir, f'ablation_layer{layer}.npz'),
            'name': f"{ckpt['folder_name']}_ablation_L{layer}",
            'itr': ckpt['itr'],
        })

    for layer in [0, 1]:
        tasks.append({
            'ckpt_path': ckpt['path'], 'type': 'cinclogits', 'layer': layer,
            'out': os.path.join(tmp_dir, f'cinclogits_layer{layer}.npz'),
            'name': f"{ckpt['folder_name']}_cinclogits_L{layer}",
            'itr': ckpt['itr'],
        })

    for ub in UB_VALUES:
        for layer in [0, 1]:
            suffix = '' if ub == 5 else f'_ub{ub}'
            tasks.append({
                'ckpt_path': ckpt['path'], 'type': 'intensity', 'layer': layer,
                'ub': ub,
                'out': os.path.join(tmp_dir, f'intensity_layer{layer}{suffix}.npz'),
                'name': f"{ckpt['folder_name']}_intensity_ub{ub}_L{layer}",
                'itr': ckpt['itr'],
            })

    return tasks


def is_checkpoint_done(ckpt):
    """Check if all tasks for a checkpoint are done."""
    tasks = make_tasks_for_checkpoint(ckpt)
    return all(os.path.exists(t['out']) for t in tasks)


def assemble_plots_for_checkpoint(ckpt):
    """Assemble all plots for a single checkpoint."""
    folder_name = ckpt['folder_name']
    tmp_dir = os.path.join(OUTPUT_BASE, 'tmp_results', folder_name)
    plot_dir = os.path.join(OUTPUT_BASE, folder_name)
    os.makedirs(plot_dir, exist_ok=True)
    tag = (f"N=256  block=16  lr=0.01  iters={ckpt['itr']}  "
           f"dseed={ckpt['dseed']}  iseed={ckpt['iseed']}")

    for fn, label in [(_assemble_baseline, 'baseline'),
                      (_assemble_ablation, 'ablation'),
                      (_assemble_cinclogits, 'cinclogits'),
                      (_assemble_intensity, 'intensity')]:
        try:
            fn(tmp_dir, plot_dir, tag)
        except Exception as e:
            print(f"  WARN {label} for {folder_name}: {e}", flush=True)

    n_plots = len([f for f in os.listdir(plot_dir) if f.endswith('.png')])
    return n_plots


def monitor_workers(procs, checkpoints, ckpt_by_gpu):
    """Monitor worker processes and assemble plots as checkpoints complete."""
    assembled = set()
    total_tasks = sum(len(make_tasks_for_checkpoint(c)) for c in checkpoints)
    total_done = 0

    # Assemble already-cached checkpoints
    for ckpt in checkpoints:
        if is_checkpoint_done(ckpt) and ckpt['folder_name'] not in assembled:
            n = assemble_plots_for_checkpoint(ckpt)
            assembled.add(ckpt['folder_name'])
            print(f"  [PLOTS] {ckpt['folder_name']}: {n} plots (cached)", flush=True)

    t0 = time.time()
    while any(p.poll() is None for p in procs.values()):
        time.sleep(5)

        for ckpt in checkpoints:
            fn = ckpt['folder_name']
            if fn not in assembled and is_checkpoint_done(ckpt):
                n = assemble_plots_for_checkpoint(ckpt)
                assembled.add(fn)
                elapsed = time.time() - t0
                print(f"  [PLOTS] {fn}: {n} plots assembled ({elapsed:.0f}s)",
                      flush=True)

        done_now = sum(1 for c in checkpoints
                       for t in make_tasks_for_checkpoint(c)
                       if os.path.exists(t['out']))
        if done_now > total_done + 10:
            total_done = done_now
            elapsed = time.time() - t0
            rate = total_done / elapsed if elapsed > 0 else 0
            eta = (total_tasks - total_done) / rate if rate > 0 else 0
            print(f"  [PROGRESS] {total_done}/{total_tasks} tasks, "
                  f"{len(assembled)}/{len(checkpoints)} ckpts plotted "
                  f"({elapsed:.0f}s, ETA ~{eta:.0f}s)", flush=True)

    # Final assembly for any remaining
    for ckpt in checkpoints:
        fn = ckpt['folder_name']
        if fn not in assembled and is_checkpoint_done(ckpt):
            n = assemble_plots_for_checkpoint(ckpt)
            assembled.add(fn)
            print(f"  [PLOTS] {fn}: {n} plots assembled (final)", flush=True)

    # Check for failures
    for gpu, proc in procs.items():
        if proc.returncode != 0:
            print(f"  [WARN] GPU {gpu} worker exited with code {proc.returncode}",
                  flush=True)

    elapsed = time.time() - t0
    return assembled, elapsed


def main():
    t_start = time.time()

    checkpoints = discover_checkpoints()
    print(f"Found {len(checkpoints)} checkpoints")

    # Filter to only checkpoints that still need work
    todo = [c for c in checkpoints if not is_checkpoint_done(c)]
    already_done = len(checkpoints) - len(todo)
    print(f"Already complete: {already_done}, to process: {len(todo)}")

    # Assemble plots for already-done checkpoints first
    for ckpt in checkpoints:
        if is_checkpoint_done(ckpt):
            plot_dir = os.path.join(OUTPUT_BASE, ckpt['folder_name'])
            if not os.path.isdir(plot_dir) or not any(f.endswith('.png') for f in os.listdir(plot_dir)):
                n = assemble_plots_for_checkpoint(ckpt)
                print(f"  [PLOTS] {ckpt['folder_name']}: {n} plots (from cache)", flush=True)

    if not todo:
        print("All checkpoints already analyzed and plotted!")
        return

    # Build task lists per GPU: distribute checkpoints round-robin
    gpu_tasks = {g: [] for g in range(NUM_GPUS)}
    for i, ckpt in enumerate(todo):
        gpu = i % NUM_GPUS
        tasks = make_tasks_for_checkpoint(ckpt)
        # Skip already-cached tasks
        tasks = [t for t in tasks if not os.path.exists(t['out'])]
        gpu_tasks[gpu].extend(tasks)

    # Sort each GPU's tasks so same-checkpoint tasks are contiguous (model loaded once)
    for g in gpu_tasks:
        gpu_tasks[g].sort(key=lambda t: (t['ckpt_path'], t['type']))

    total_tasks = sum(len(v) for v in gpu_tasks.values())
    print(f"\nTotal tasks to run: {total_tasks}, distributed across {NUM_GPUS} GPUs")
    for g in range(NUM_GPUS):
        n = len(gpu_tasks[g])
        ckpts = len(set(t['ckpt_path'] for t in gpu_tasks[g])) if n else 0
        print(f"  GPU {g}: {n} tasks across {ckpts} checkpoints")

    # Write task files and launch workers
    task_dir = os.path.join(OUTPUT_BASE, 'task_files')
    os.makedirs(task_dir, exist_ok=True)

    procs = {}
    log_dir = os.path.join(OUTPUT_BASE, 'gpu_worker_logs')
    os.makedirs(log_dir, exist_ok=True)

    for g in range(NUM_GPUS):
        if not gpu_tasks[g]:
            continue
        task_file = os.path.join(task_dir, f'gpu{g}_tasks.json')
        with open(task_file, 'w') as f:
            json.dump(gpu_tasks[g], f)

        log_file = open(os.path.join(log_dir, f'gpu{g}.log'), 'w')
        proc = subprocess.Popen(
            [sys.executable, os.path.join(SCRIPT_DIR, 'gpu_worker.py'),
             '--tasks-file', task_file, '--gpu', str(g)],
            stdout=log_file, stderr=subprocess.STDOUT,
            cwd=SCRIPT_DIR)
        procs[g] = proc

    print(f"\nLaunched {len(procs)} GPU workers. Monitoring...\n", flush=True)

    assembled, elapsed = monitor_workers(procs, checkpoints, gpu_tasks)

    total_elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"ALL DONE — {len(assembled)}/{len(checkpoints)} checkpoints plotted")
    print(f"Elapsed: {total_elapsed:.0f}s ({total_elapsed/60:.1f}m)")
    print(f"Output: {OUTPUT_BASE}")
    print(f"{'='*60}")


# ── Plot assembly functions ──────────────────────────────────────────────────

def _assemble_baseline(tmp_dir, plot_dir, tag):
    f = os.path.join(tmp_dir, 'baseline.npz')
    if not os.path.exists(f):
        return
    d = np.load(f)
    full_seq_acc = float(d['full_seq_acc'])
    cond_acc = d['cond_acc']
    cond_eligible = d['cond_eligible']

    fig, ax = plt.subplots(figsize=(4, 4))
    bars = ax.bar([0], [full_seq_acc], 0.5, color='#e6850e')
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + 0.01,
                f'{h:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_xticks([0])
    ax.set_xticklabels(['Model'], fontsize=12)
    ax.set_ylabel('Full-sequence accuracy', fontsize=12)
    ax.set_title(f'Baseline accuracy (500 trials)\n{tag}', fontsize=11, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.2, linestyle=':')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0, min(1.15, full_seq_acc * 1.2 + 0.05))
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, 'baseline_accuracy.png'), dpi=300, bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(6, 4))
    pos = np.arange(len(cond_acc))
    valid = cond_eligible >= 10
    if valid.any():
        ax.plot(pos[valid], cond_acc[valid], marker='s', markersize=3, linewidth=1.2,
                color='#e6850e')
        if not valid.all():
            cutoff = np.where(~valid)[0][0]
            ax.axvline(x=cutoff - 0.5, color='#e6850e', linestyle=':', alpha=0.5)
    ax.set_xlabel('Output position', fontsize=10)
    ax.set_ylabel('Conditional per-token accuracy', fontsize=10)
    ax.set_title(f'Per-token accuracy (given correct prefix) — baseline (500 trials)\n{tag}',
                 fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, 'baseline_conditional_accuracy.png'),
                dpi=300, bbox_inches='tight')
    plt.close()


def _assemble_ablation(tmp_dir, plot_dir, tag):
    data = {}
    for layer in [0, 1]:
        f = os.path.join(tmp_dir, f'ablation_layer{layer}.npz')
        if not os.path.exists(f):
            return
        d = np.load(f)
        data[layer] = {
            'full_seq_acc': float(d['full_seq_acc']),
            'per_pos_acc': d['per_pos_acc'],
            'cond_acc': d['cond_acc'],
            'cond_eligible': d['cond_eligible'],
        }

    fig, ax = plt.subplots(figsize=(5, 4.5))
    vals = [data[0]['full_seq_acc'], data[1]['full_seq_acc']]
    bars = ax.bar([0, 1], vals, 0.5, color=['#1f77b4', '#ff7f0e'])
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + 0.01,
                f'{h:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Skip Layer 0', 'Skip Layer 1'], fontsize=12)
    ax.set_ylabel('Full-sequence accuracy', fontsize=12)
    ax.set_title(f'Accuracy with attention layer removed (500 trials)\n{tag}',
                 fontsize=11, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.2, linestyle=':')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0, min(1.15, max(vals) * 1.25 + 0.05))
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, 'ablation_accuracy.png'), dpi=300, bbox_inches='tight')
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for i, layer in enumerate([0, 1]):
        ax = axes[i]
        pos = np.arange(len(data[layer]['per_pos_acc']))
        ax.plot(pos, data[layer]['per_pos_acc'], marker='o', markersize=3,
                linewidth=1.2, color=['#1f77b4', '#ff7f0e'][i])
        ax.set_xlabel('Output position', fontsize=10)
        ax.set_title(f'Skip Layer {layer}', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel('Per-position accuracy', fontsize=10)
    fig.suptitle(f'Per-position accuracy with attention removed (500 trials)\n{tag}',
                 fontsize=11, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, 'ablation_per_position.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for i, layer in enumerate([0, 1]):
        ax = axes[i]
        ca = data[layer]['cond_acc']
        ce = data[layer]['cond_eligible']
        pos = np.arange(len(ca))
        valid = ce >= 10
        color = ['#1f77b4', '#ff7f0e'][i]
        if valid.any():
            ax.plot(pos[valid], ca[valid], marker='o', markersize=3,
                    linewidth=1.2, color=color)
            if not valid.all():
                cutoff = np.where(~valid)[0][0]
                ax.axvline(x=cutoff - 0.5, color=color, linestyle=':', alpha=0.5)
        ax.set_xlabel('Output position', fontsize=10)
        ax.set_title(f'Skip Layer {layer}', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel('Conditional per-token accuracy', fontsize=10)
    fig.suptitle(
        f'Per-token accuracy (given prefix correct) with attention removed (500 trials)\n{tag}',
        fontsize=11, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, 'ablation_conditional_accuracy.png'),
                dpi=300, bbox_inches='tight')
    plt.close()


def _assemble_cinclogits(tmp_dir, plot_dir, tag):
    for layer in [0, 1]:
        f = os.path.join(tmp_dir, f'cinclogits_layer{layer}.npz')
        if not os.path.exists(f):
            continue
        d = np.load(f)
        cl_ic = d['clogit_icscore']
        icl_ic = d['iclogit_icscore']
        frac_ic = np.mean(cl_ic + icl_ic)
        eps = 1e-10
        corr = np.sum(cl_ic) / (np.sum(cl_ic + icl_ic) + eps)

        fig, ax = plt.subplots(figsize=(4.5, 4))
        bw = 0.5
        x = np.array([0, 1])
        b1 = ax.bar(x[0], frac_ic, bw, color='#e6850e')
        b2 = ax.bar(x[1], corr, bw, color='#1f77b4')
        for bars in [b1, b2]:
            for b in bars:
                h = b.get_height()
                ax.text(b.get_x() + b.get_width() / 2, h + 0.008,
                        f'{h:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(['Fraction of\nincorrect scores',
                            'Logit correction ratio\namong incorrect scores'], fontsize=10)
        ax.set_ylabel('Fraction', fontsize=12)
        ax.set_title(f'Incorrect scores & logit correction (Layer {layer})\n{tag}',
                     fontsize=11, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.2, linestyle=':')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ymax = max(frac_ic, corr)
        ax.set_ylim(0, ymax * 1.25 if ymax > 0 else 1.0)
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, f'cinclogits_layer{layer}.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()


def _assemble_intensity(tmp_dir, plot_dir, tag):
    for ub in UB_VALUES:
        for layer in [0, 1]:
            suffix = '' if ub == 5 else f'_ub{ub}'
            f = os.path.join(tmp_dir, f'intensity_layer{layer}{suffix}.npz')
            if not os.path.exists(f):
                continue
            d = np.load(f)
            intensities = d['intensities']
            rates = d['success_rates']

            plt.figure(figsize=(4.5, 3.2))
            plt.plot(intensities, rates, marker='o', linewidth=1.5, markersize=5,
                     color='#e6850e')
            plt.xlabel('Intervention Intensity', fontsize=9)
            plt.ylabel('Success Probability', fontsize=9)
            title = f'Robustness to Attention Intervention (Layer {layer})'
            if ub != 5:
                title += f'  [ub={ub}]'
            title += f'\n{tag}'
            plt.title(title, fontsize=9)
            plt.grid(True, alpha=0.3)
            plt.xticks(intensities, fontsize=8)
            plt.yticks(fontsize=8)
            plt.ylim(0, 1.05)
            plt.tight_layout()
            fname = f'intensity_layer{layer}'
            if ub != 5:
                fname += f'_ub{ub}'
            plt.savefig(os.path.join(plot_dir, f'{fname}.png'),
                        dpi=300, bbox_inches='tight')
            plt.close()


if __name__ == '__main__':
    main()
