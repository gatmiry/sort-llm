"""
Parallel analysis launcher for grid-run-2 individual checkpoints.
Unlike grid-run which requires LN0+LN1 pairs, this analyzes each completed
checkpoint independently and produces per-model plots.

Dispatches fine-grained worker processes (one per checkpoint x task x layer)
across GPUs, then assembles plots into folders matching grid-run's format:
  plots_{run_name}_ckpt{itr}/

Plots produced per checkpoint:
  - cinclogits_layer{0,1}.png
  - intensity_layer{0,1}.png
  - intensity_layer{0,1}_ub{10,15}.png
  - ablation_accuracy.png
  - ablation_per_position.png
  - ablation_conditional_accuracy.png
  - baseline_accuracy.png
  - baseline_conditional_accuracy.png
"""
import argparse
import os
import re
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
GPU_MEM_TOTAL = 81559
MEM_CAP_FRAC = 0.90
MAX_WORKERS_PER_GPU = 10

WORKER_MEM_EST = {64: 200, 128: 300, 256: 400, 512: 600, 8192: 2000}

UB_VALUES = [10, 15, 20, 30]


def get_gpu_mem_used():
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'],
            text=True)
        usage = {}
        for line in out.strip().split('\n'):
            parts = line.split(',')
            usage[int(parts[0].strip())] = int(parts[1].strip())
        return usage
    except Exception:
        return {i: 0 for i in range(NUM_GPUS)}


def find_completed_runs():
    """Find all run directories with a final checkpoint."""
    runs = []
    if not os.path.isdir(OUTPUT_BASE):
        return runs
    pattern = re.compile(
        r'^V(\d+)_B(\d+)_LR([\d.e+-]+)_MI(\d+)_LN(\d)_E64_H1_L2$')
    for d in sorted(os.listdir(OUTPUT_BASE)):
        dirpath = os.path.join(OUTPUT_BASE, d)
        if not os.path.isdir(dirpath):
            continue
        m = pattern.match(d)
        if not m:
            continue
        vs, bs, lr, mi, ln = int(m.group(1)), int(m.group(2)), m.group(3), int(m.group(4)), int(m.group(5))
        target = f"_itr{mi}.pt"
        ckpt = None
        for f in os.listdir(dirpath):
            if f.endswith(target):
                ckpt = os.path.join(dirpath, f)
                break
        if ckpt is None:
            continue
        runs.append({
            'vocab_size': vs, 'block_size': bs, 'lr': lr, 'max_iters': mi,
            'ln': ln, 'run_name': d, 'ckpt': ckpt,
        })
    return runs


def make_tasks(runs):
    """Generate all worker tasks for all completed runs."""
    tasks = []
    for run in runs:
        vs = run['vocab_size']
        tmp_dir = os.path.join(OUTPUT_BASE, 'tmp_results', run['run_name'])
        for task_type in ['cinclogits', 'intensity', 'ablation']:
            for layer in [0, 1]:
                out_file = os.path.join(tmp_dir, f"{task_type}_layer{layer}.npz")
                tasks.append({
                    'run': run, 'ckpt': run['ckpt'],
                    'task': task_type, 'layer': layer,
                    'out': out_file,
                    'name': f"{run['run_name']}_{task_type}_L{layer}",
                    'est_mem': WORKER_MEM_EST.get(vs, 500),
                })
        for ub in UB_VALUES:
            for layer in [0, 1]:
                out_file = os.path.join(tmp_dir, f"intensity_layer{layer}_ub{ub}.npz")
                tasks.append({
                    'run': run, 'ckpt': run['ckpt'],
                    'task': 'intensity', 'layer': layer,
                    'unsorted_lb': ub, 'unsorted_ub': ub,
                    'out': out_file,
                    'name': f"{run['run_name']}_intensity_ub{ub}_L{layer}",
                    'est_mem': WORKER_MEM_EST.get(vs, 500),
                })
        out_file = os.path.join(tmp_dir, "baseline.npz")
        tasks.append({
            'run': run, 'ckpt': run['ckpt'],
            'task': 'baseline', 'layer': 0,
            'out': out_file,
            'name': f"{run['run_name']}_baseline",
            'est_mem': WORKER_MEM_EST.get(vs, 500),
        })
    return tasks


def is_task_done(task):
    return os.path.exists(task['out'])


def launch_worker(task, gpu_id):
    cmd = [
        sys.executable, os.path.join(SCRIPT_DIR, 'analyze_worker.py'),
        '--ckpt', task['ckpt'],
        '--task', task['task'],
        '--layer', str(task['layer']),
        '--out', task['out'],
        '--device', 'cuda',
    ]
    if 'unsorted_lb' in task:
        cmd += ['--unsorted_lb', str(task['unsorted_lb']),
                '--unsorted_ub', str(task['unsorted_ub'])]
    log_dir = os.path.join(OUTPUT_BASE, 'analysis_logs')
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{task['name']}.log")
    log_file = open(log_path, 'w')
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT,
                            env=env, cwd=SCRIPT_DIR)
    return proc, log_file


def run_workers(tasks_to_run):
    if not tasks_to_run:
        print("Nothing to run.")
        return
    queue = list(tasks_to_run)
    gpu_jobs = {g: [] for g in range(NUM_GPUS)}
    gpu_analysis_mem = {g: 0 for g in range(NUM_GPUS)}
    completed = 0
    failed = 0
    total = len(tasks_to_run)
    t0 = time.time()

    while queue or any(gpu_jobs[g] for g in range(NUM_GPUS)):
        for g in range(NUM_GPUS):
            still_running = []
            for proc, log_file, name, mem in gpu_jobs[g]:
                ret = proc.poll()
                if ret is not None:
                    log_file.close()
                    gpu_analysis_mem[g] -= mem
                    if ret == 0:
                        completed += 1
                        if completed % 20 == 0 or completed == total:
                            elapsed = time.time() - t0
                            print(f"[PROGRESS] {completed}/{total} done, {failed} fail, "
                                  f"{len(queue)} queued ({elapsed:.0f}s)")
                    else:
                        failed += 1
                        print(f"[FAIL] GPU {g}: {name} exit={ret}")
                else:
                    still_running.append((proc, log_file, name, mem))
            gpu_jobs[g] = still_running

        mem_used = get_gpu_mem_used()

        remaining = []
        for task in queue:
            best_gpu = None
            best_headroom = -1
            for g in range(NUM_GPUS):
                if len(gpu_jobs[g]) >= MAX_WORKERS_PER_GPU:
                    continue
                cap = GPU_MEM_TOTAL * MEM_CAP_FRAC
                headroom = cap - mem_used.get(g, 0) - gpu_analysis_mem[g]
                if headroom >= task['est_mem'] and headroom > best_headroom:
                    best_gpu = g
                    best_headroom = headroom
            if best_gpu is not None:
                proc, log_file = launch_worker(task, best_gpu)
                gpu_jobs[best_gpu].append((proc, log_file, task['name'], task['est_mem']))
                gpu_analysis_mem[best_gpu] += task['est_mem']
            else:
                remaining.append(task)
        queue = remaining

        if any(gpu_jobs[g] for g in range(NUM_GPUS)):
            time.sleep(2)

    elapsed = time.time() - t0
    print(f"\nAll workers done! completed={completed}, failed={failed}, elapsed={elapsed:.0f}s")


# ── Plot assembly ────────────────────────────────────────────────────────────

def _load(tmp_dir, name):
    return np.load(os.path.join(tmp_dir, name))


def assemble_all_plots(run):
    rn = run['run_name']
    tmp_dir = os.path.join(OUTPUT_BASE, 'tmp_results', rn)
    ln_label = 'With LayerNorm' if run['ln'] == 1 else 'Without LayerNorm'
    color = '#e6850e' if run['ln'] == 1 else '#6a3d9a'

    try:
        itr = int(_load(tmp_dir, 'cinclogits_layer0.npz')['itr'])
    except Exception:
        itr = run['max_iters']

    tag = (f"vocab={run['vocab_size']}  block={run['block_size']}  lr={run['lr']}  "
           f"iters={run['max_iters']}  LN={run['ln']}  ckpt={itr}")

    plot_dir = os.path.join(OUTPUT_BASE, f"plots_{rn}_ckpt{itr}")
    os.makedirs(plot_dir, exist_ok=True)

    _assemble_cinclogits(tmp_dir, plot_dir, tag, ln_label, color)
    _assemble_intensity(tmp_dir, plot_dir, tag, ln_label, color)
    _assemble_intensity_ub(tmp_dir, plot_dir, tag, ln_label, color)
    _assemble_ablation(tmp_dir, plot_dir, tag, ln_label, color)
    _assemble_baseline(tmp_dir, plot_dir, tag, ln_label, color)

    print(f"  Plots saved to {plot_dir}")


def _assemble_cinclogits(tmp_dir, plot_dir, tag, ln_label, color):
    for layer in [0, 1]:
        f = os.path.join(tmp_dir, f'cinclogits_layer{layer}.npz')
        if not os.path.exists(f):
            continue
        r = np.load(f)
        cl_ic, icl_ic = r['clogit_icscore'], r['iclogit_icscore']
        frac_ic = np.mean(cl_ic + icl_ic)
        eps = 1e-10
        corr = np.sum(cl_ic) / (np.sum(cl_ic + icl_ic) + eps)

        fig, ax = plt.subplots(figsize=(5, 4.2))
        bw = 0.4
        x = np.array([0, 1])
        b1 = ax.bar(x[0], frac_ic, bw, color=color, label=ln_label)
        b2 = ax.bar(x[1], corr, bw, color=color)
        for bar in [b1, b2]:
            for b in bar:
                h = b.get_height()
                ax.text(b.get_x() + b.get_width() / 2, h + 0.008,
                        f'{h:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(['Fraction of\nincorrect scores',
                            'Logit correction ratio\namong incorrect scores'], fontsize=11)
        ax.set_ylabel('Fraction', fontsize=12)
        title = f'Incorrect scores & logit correction (Layer {layer})'
        if tag:
            title += f'\n{tag}'
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend(fontsize=10, framealpha=0.9)
        ax.grid(True, axis='y', alpha=0.2, linestyle=':')
        ax.tick_params(labelsize=11)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ymax = max(frac_ic, corr)
        ax.set_ylim(0, ymax * 1.25 if ymax > 0 else 1.0)
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, f'cinclogits_layer{layer}.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()


def _assemble_intensity(tmp_dir, plot_dir, tag, ln_label, color, ub_suffix=""):
    for layer in [0, 1]:
        fname = f'intensity_layer{layer}.npz' if not ub_suffix else f'intensity_layer{layer}_ub{ub_suffix}.npz'
        f = os.path.join(tmp_dir, fname)
        if not os.path.exists(f):
            continue
        r = np.load(f)
        intensities, rates = r['intensities'], r['success_rates']

        plt.figure(figsize=(3.5, 2.8))
        plt.plot(intensities, rates, marker='o', linewidth=1.5, markersize=5,
                 label=ln_label.lower(), color=color)
        plt.xlabel('Intervention Intensity', fontsize=9)
        plt.ylabel('Success Probability', fontsize=9)
        title = f'Robustness to Attention Intervention (Layer {layer})'
        if ub_suffix:
            title += f'  [ub={ub_suffix}]'
        if tag:
            title += f'\n{tag}'
        plt.title(title, fontsize=10)
        plt.legend(fontsize=7, loc='lower left')
        plt.grid(True, alpha=0.3)
        plt.xticks(list(intensities[::2]), fontsize=8)
        plt.yticks(fontsize=8)
        plt.ylim(0, 1.05)
        plt.tight_layout()
        out_name = f'intensity_layer{layer}'
        if ub_suffix:
            out_name += f'_ub{ub_suffix}'
        plt.savefig(os.path.join(plot_dir, f'{out_name}.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()


def _assemble_intensity_ub(tmp_dir, plot_dir, tag, ln_label, color):
    for ub in UB_VALUES:
        _assemble_intensity(tmp_dir, plot_dir, tag, ln_label, color, ub_suffix=str(ub))


def _assemble_ablation(tmp_dir, plot_dir, tag, ln_label, color):
    data = {}
    for layer in [0, 1]:
        f = os.path.join(tmp_dir, f'ablation_layer{layer}.npz')
        if not os.path.exists(f):
            return
        d = np.load(f)
        data[layer] = {
            'full_seq_acc': float(d['full_seq_acc']),
            'per_pos_acc': d['per_pos_acc'],
            'cond_acc': d.get('cond_acc', None),
            'cond_eligible': d.get('cond_eligible', None),
        }

    fig, ax = plt.subplots(figsize=(5, 4))
    labels = ['Skip Layer 0', 'Skip Layer 1']
    x = np.arange(len(labels))
    bw = 0.4
    vals = [data[0]['full_seq_acc'], data[1]['full_seq_acc']]
    bars = ax.bar(x, vals, bw, color=color, label=ln_label)
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + 0.01,
                f'{h:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel('Full-sequence accuracy', fontsize=12)
    title = 'Accuracy with attention layer removed (500 trials)'
    if tag:
        title += f'\n{tag}'
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(True, axis='y', alpha=0.2, linestyle=':')
    ax.tick_params(labelsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0, min(1.15, max(vals) * 1.25 + 0.05))
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, 'ablation_accuracy.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for i, layer in enumerate([0, 1]):
        ax = axes[i]
        pos = np.arange(len(data[layer]['per_pos_acc']))
        ax.plot(pos, data[layer]['per_pos_acc'],
                marker='o', markersize=3, linewidth=1.2,
                label=ln_label, color=color)
        ax.set_xlabel('Output position', fontsize=10)
        ax.set_title(f'Skip Layer {layer}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel('Per-position accuracy', fontsize=10)
    fig.suptitle(f'Per-position accuracy with attention removed (500 trials)\n{tag}',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, 'ablation_per_position.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    if data[0].get('cond_acc') is None:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for i, layer in enumerate([0, 1]):
        ax = axes[i]
        cond_acc = data[layer]['cond_acc']
        cond_elig = data[layer]['cond_eligible']
        pos = np.arange(len(cond_acc))
        valid = cond_elig >= 10
        ax.plot(pos[valid], cond_acc[valid],
                marker='o', markersize=3, linewidth=1.2,
                label=ln_label, color=color)
        if not valid.all():
            cutoff = np.where(~valid)[0]
            if len(cutoff) > 0:
                ax.axvline(x=cutoff[0] - 0.5, color=color, linestyle=':', alpha=0.5)
        ax.set_xlabel('Output position', fontsize=10)
        ax.set_title(f'Skip Layer {layer}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
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


def _assemble_baseline(tmp_dir, plot_dir, tag, ln_label, color):
    f = os.path.join(tmp_dir, 'baseline.npz')
    if not os.path.exists(f):
        return
    d = np.load(f)
    full_seq_acc = float(d['full_seq_acc'])
    per_pos_acc = d['per_pos_acc']
    cond_acc = d['cond_acc']
    cond_elig = d['cond_eligible']

    fig, ax = plt.subplots(figsize=(4, 4))
    bw = 0.4
    bar = ax.bar([0], [full_seq_acc], bw, color=color)
    for b in bar:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + 0.01,
                f'{h:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_xticks([0])
    ax.set_xticklabels([ln_label], fontsize=11)
    ax.set_ylabel('Full-sequence accuracy', fontsize=12)
    title = 'Baseline accuracy (500 trials)'
    if tag:
        title += f'\n{tag}'
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.2, linestyle=':')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0, min(1.15, full_seq_acc * 1.2 + 0.05))
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, 'baseline_accuracy.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(6, 4))
    pos = np.arange(len(cond_acc))
    valid = cond_elig >= 10
    ax.plot(pos[valid], cond_acc[valid],
            marker='o', markersize=3, linewidth=1.2,
            label=ln_label, color=color)
    ax.set_xlabel('Output position', fontsize=10)
    ax.set_ylabel('Conditional per-token accuracy', fontsize=10)
    title = 'Per-token accuracy (given correct prefix) — baseline (500 trials)'
    if tag:
        title += f'\n{tag}'
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, 'baseline_conditional_accuracy.png'),
                dpi=300, bbox_inches='tight')
    plt.close()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    runs = find_completed_runs()
    print(f"Found {len(runs)} completed runs with final checkpoints")
    if not runs:
        print("Nothing to analyze.")
        return

    for r in runs:
        print(f"  {r['run_name']}  (ckpt: {os.path.basename(r['ckpt'])})")

    all_tasks = make_tasks(runs)
    tasks_to_run = [t for t in all_tasks if not is_task_done(t)]
    print(f"\nTotal worker tasks: {len(all_tasks)}, "
          f"cached: {len(all_tasks) - len(tasks_to_run)}, "
          f"to run: {len(tasks_to_run)}")

    if args.dry_run:
        for t in tasks_to_run[:30]:
            print(f"  {t['name']}")
        if len(tasks_to_run) > 30:
            print(f"  ... and {len(tasks_to_run) - 30} more")
        return

    tasks_to_run.sort(key=lambda t: t['run']['vocab_size'])
    run_workers(tasks_to_run)

    print("\nAssembling plots...")
    for run in runs:
        try:
            assemble_all_plots(run)
        except Exception as e:
            print(f"  ERROR assembling {run['run_name']}: {e}")

    print("\nDone!")


if __name__ == '__main__':
    main()
