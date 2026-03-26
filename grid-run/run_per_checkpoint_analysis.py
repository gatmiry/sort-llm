"""
Generate analysis plots for EVERY checkpoint of a given model config,
organized in per-checkpoint subfolders.

Dispatches analyze_worker.py across all available GPUs.

Usage:
    python run_per_checkpoint_analysis.py
"""
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
ARCH_SUFFIX = 'E64_H1_L2'
NUM_GPUS = 8
MAX_WORKERS_PER_GPU = 6

def _parse_config():
    import argparse as _ap
    p = _ap.ArgumentParser()
    p.add_argument('--config', default='V128_B32_LR1e-03_MI40000',
                   help='Config base, e.g. V256_B32_LR1e-02_MI40000')
    return p.parse_args()

_ARGS = _parse_config()
CONFIG_BASE = _ARGS.config

_mi = re.search(r'MI(\d+)', CONFIG_BASE).group(1)
PARENT_PLOT_DIR = os.path.join(
    OUTPUT_BASE, f'plots_{CONFIG_BASE}_{ARCH_SUFFIX}_ckpt{_mi}', 'per_checkpoint'
)

LN0_DIR = os.path.join(OUTPUT_BASE, f'{CONFIG_BASE}_LN0_{ARCH_SUFFIX}')
LN1_DIR = os.path.join(OUTPUT_BASE, f'{CONFIG_BASE}_LN1_{ARCH_SUFFIX}')


def find_checkpoint_iterations():
    """Find iteration numbers present in BOTH LN0 and LN1 dirs."""
    pat = re.compile(r'_itr(\d+)\.pt$')
    def get_iters(d):
        return {int(pat.search(f).group(1)) for f in os.listdir(d)
                if f.endswith('.pt') and pat.search(f)}
    iters_0 = get_iters(LN0_DIR)
    iters_1 = get_iters(LN1_DIR)
    return sorted(iters_0 & iters_1)


def ckpt_path(ln_dir, itr):
    base = os.path.basename(ln_dir)
    return os.path.join(ln_dir, f'{base}_itr{itr}.pt')


def build_tasks(iterations):
    """Build all worker tasks: cinclogits, intensity (ub5, ub60), ablation, baseline."""
    tasks = []
    for itr in iterations:
        tmp_dir = os.path.join(OUTPUT_BASE, 'tmp_perckpt', f'{CONFIG_BASE}_itr{itr}')
        for ln_label, ln_dir in [('LN0', LN0_DIR), ('LN1', LN1_DIR)]:
            ckpt = ckpt_path(ln_dir, itr)
            for layer in [0, 1]:
                tasks.append({
                    'itr': itr, 'ckpt': ckpt, 'ln': ln_label,
                    'task': 'cinclogits', 'layer': layer,
                    'out': os.path.join(tmp_dir, f'cinclogits_{ln_label}_layer{layer}.npz'),
                    'name': f'itr{itr}_cinclogits_{ln_label}_L{layer}',
                    'extra_args': [],
                })
                tasks.append({
                    'itr': itr, 'ckpt': ckpt, 'ln': ln_label,
                    'task': 'intensity', 'layer': layer,
                    'out': os.path.join(tmp_dir, f'intensity_{ln_label}_layer{layer}.npz'),
                    'name': f'itr{itr}_intensity_{ln_label}_L{layer}',
                    'extra_args': [],
                })
                tasks.append({
                    'itr': itr, 'ckpt': ckpt, 'ln': ln_label,
                    'task': 'intensity', 'layer': layer,
                    'out': os.path.join(tmp_dir, f'intensity_{ln_label}_layer{layer}_ub60.npz'),
                    'name': f'itr{itr}_intensity_ub60_{ln_label}_L{layer}',
                    'extra_args': ['--unsorted_lb', '60', '--unsorted_ub', '60'],
                })
                tasks.append({
                    'itr': itr, 'ckpt': ckpt, 'ln': ln_label,
                    'task': 'ablation', 'layer': layer,
                    'out': os.path.join(tmp_dir, f'ablation_{ln_label}_layer{layer}.npz'),
                    'name': f'itr{itr}_ablation_{ln_label}_L{layer}',
                    'extra_args': [],
                })
            tasks.append({
                'itr': itr, 'ckpt': ckpt, 'ln': ln_label,
                'task': 'baseline', 'layer': 0,
                'out': os.path.join(tmp_dir, f'baseline_{ln_label}.npz'),
                'name': f'itr{itr}_baseline_{ln_label}',
                'extra_args': [],
            })
    return tasks


def launch_worker(task, gpu_id):
    cmd = [
        sys.executable, os.path.join(SCRIPT_DIR, 'analyze_worker.py'),
        '--ckpt', task['ckpt'],
        '--task', task['task'],
        '--layer', str(task['layer']),
        '--out', task['out'],
        '--device', 'cuda',
    ] + task['extra_args']
    os.makedirs(os.path.dirname(task['out']), exist_ok=True)
    log_dir = os.path.join(OUTPUT_BASE, 'analysis_logs')
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f'perckpt_{task["name"]}.log')
    log_file = open(log_path, 'w')
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT,
                            env=env, cwd=SCRIPT_DIR)
    return proc, log_file


def run_workers(tasks_to_run):
    gpu_jobs = {g: [] for g in range(NUM_GPUS)}
    queue = list(tasks_to_run)
    completed = 0
    failed = 0
    total = len(tasks_to_run)
    t0 = time.time()

    while queue or any(gpu_jobs[g] for g in range(NUM_GPUS)):
        for g in range(NUM_GPUS):
            still = []
            for proc, lf, name in gpu_jobs[g]:
                ret = proc.poll()
                if ret is not None:
                    lf.close()
                    if ret == 0:
                        completed += 1
                    else:
                        failed += 1
                        print(f'[FAIL] GPU {g}: {name} exit={ret}')
                else:
                    still.append((proc, lf, name))
            gpu_jobs[g] = still

        if completed % 40 == 0 or not queue:
            elapsed = time.time() - t0
            running = sum(len(gpu_jobs[g]) for g in range(NUM_GPUS))
            print(f'[{elapsed:6.0f}s] done={completed}/{total}  fail={failed}  '
                  f'running={running}  queued={len(queue)}', flush=True)

        placed = []
        for i, task in enumerate(queue):
            best_gpu = None
            best_load = MAX_WORKERS_PER_GPU + 1
            for g in range(NUM_GPUS):
                if len(gpu_jobs[g]) < MAX_WORKERS_PER_GPU and len(gpu_jobs[g]) < best_load:
                    best_gpu = g
                    best_load = len(gpu_jobs[g])
            if best_gpu is not None:
                proc, lf = launch_worker(task, best_gpu)
                gpu_jobs[best_gpu].append((proc, lf, task['name']))
                placed.append(i)
            else:
                break
        for i in reversed(placed):
            queue.pop(i)

        time.sleep(1)

    elapsed = time.time() - t0
    print(f'\nAll workers done: completed={completed}, failed={failed}, elapsed={elapsed:.0f}s')


# ── Plotting (reuses same style as run_all_analysis.py) ──────────────────────

def _plot_cinclogits(r0, r1, plot_dir, layer, tag):
    cl0, icl0 = r0['clogit_icscore'], r0['iclogit_icscore']
    cl1, icl1 = r1['clogit_icscore'], r1['iclogit_icscore']
    frac0 = np.mean(cl0 + icl0)
    frac1 = np.mean(cl1 + icl1)
    eps = 1e-10
    corr0 = np.sum(cl0) / (np.sum(cl0 + icl0) + eps)
    corr1 = np.sum(cl1) / (np.sum(cl1 + icl1) + eps)

    fig, ax = plt.subplots(figsize=(5, 4.2))
    bw = 0.32
    x = np.array([0, 1])
    b1 = ax.bar(x[0] - bw/2, frac0, bw, color='#6a3d9a', label='Without LayerNorm')
    b2 = ax.bar(x[0] + bw/2, frac1, bw, color='#e6850e', label='With LayerNorm')
    b3 = ax.bar(x[1] - bw/2, corr0, bw, color='#6a3d9a')
    b4 = ax.bar(x[1] + bw/2, corr1, bw, color='#e6850e')
    for bar in [b1, b2, b3, b4]:
        for b in bar:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width()/2, h + 0.008,
                    f'{h:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Fraction of\nincorrect scores',
                        'Logit correction ratio\namong incorrect scores'], fontsize=11)
    ax.set_ylabel('Fraction', fontsize=12)
    ax.set_title(f'Incorrect scores & logit correction (Layer {layer})\n{tag}',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(True, axis='y', alpha=0.2, linestyle=':')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ymax = max(frac0, frac1, corr0, corr1)
    ax.set_ylim(0, ymax * 1.25 if ymax > 0 else 1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, f'compare_cinclogits_layer{layer}.png'),
                dpi=300, bbox_inches='tight')
    plt.close()


def _plot_intensity(i0, i1, layer, plot_dir, tag, ub_suffix=''):
    plt.figure(figsize=(3.5, 2.8))
    plt.plot(i0['intensities'], i0['success_rates'], marker='o', linewidth=1.5,
             markersize=5, label='without layer norm', color='#1f77b4')
    plt.plot(i1['intensities'], i1['success_rates'], marker='s', linewidth=1.5,
             markersize=5, label='with layer norm', color='#ff7f0e')
    plt.xlabel('Intervention Intensity', fontsize=9)
    plt.ylabel('Success Probability', fontsize=9)
    title = f'Robustness to Attention Intervention (Layer {layer})'
    if ub_suffix:
        title += f'  [ub={ub_suffix}]'
    title += f'\n{tag}'
    plt.title(title, fontsize=10)
    plt.legend(fontsize=7, loc='lower left')
    plt.grid(True, alpha=0.3)
    plt.xticks(list(i0['intensities'][::2]), fontsize=8)
    plt.yticks(fontsize=8)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    fname = f'compare_intensity_layer{layer}'
    if ub_suffix:
        fname += f'_ub{ub_suffix}'
    plt.savefig(os.path.join(plot_dir, f'{fname}.png'), dpi=300, bbox_inches='tight')
    plt.close()


def _plot_ablation(data, plot_dir, tag):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    labels = ['Skip Layer 0', 'Skip Layer 1']
    x = np.arange(len(labels))
    bw = 0.3
    ln0_vals = [float(data[('LN0', 0)]['full_seq_acc']), float(data[('LN0', 1)]['full_seq_acc'])]
    ln1_vals = [float(data[('LN1', 0)]['full_seq_acc']), float(data[('LN1', 1)]['full_seq_acc'])]
    b1 = ax.bar(x - bw/2, ln0_vals, bw, color='#6a3d9a', label='Without LayerNorm')
    b2 = ax.bar(x + bw/2, ln1_vals, bw, color='#e6850e', label='With LayerNorm')
    for bars in [b1, b2]:
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width()/2, h + 0.01,
                    f'{h:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel('Full-sequence accuracy', fontsize=12)
    ax.set_title(f'Accuracy with attention layer removed (500 trials)\n{tag}',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(True, axis='y', alpha=0.2, linestyle=':')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0, min(1.15, max(ln0_vals + ln1_vals) * 1.25 + 0.05))
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, 'ablation_accuracy.png'), dpi=300, bbox_inches='tight')
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for i, layer in enumerate([0, 1]):
        ax = axes[i]
        pos = np.arange(len(data[('LN0', layer)]['per_pos_acc']))
        ax.plot(pos, data[('LN0', layer)]['per_pos_acc'], marker='o', markersize=3,
                linewidth=1.2, label='Without LayerNorm', color='#6a3d9a')
        ax.plot(pos, data[('LN1', layer)]['per_pos_acc'], marker='s', markersize=3,
                linewidth=1.2, label='With LayerNorm', color='#e6850e')
        ax.set_xlabel('Output position', fontsize=10)
        ax.set_title(f'Skip Layer {layer}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel('Per-position accuracy', fontsize=10)
    fig.suptitle(f'Per-position accuracy with attention removed (500 trials)\n{tag}',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, 'ablation_per_position.png'), dpi=300, bbox_inches='tight')
    plt.close()

    if 'cond_acc' not in data[('LN0', 0)]:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for i, layer in enumerate([0, 1]):
        ax = axes[i]
        for ln, color, marker, label in [
            ('LN0', '#6a3d9a', 'o', 'Without LayerNorm'),
            ('LN1', '#e6850e', 's', 'With LayerNorm'),
        ]:
            ca = data[(ln, layer)]['cond_acc']
            ce = data[(ln, layer)]['cond_eligible']
            pos = np.arange(len(ca))
            valid = ce >= 10
            ax.plot(pos[valid], ca[valid], marker=marker, markersize=3,
                    linewidth=1.2, label=label, color=color)
            if not valid.all() and valid.any():
                cutoff = np.where(~valid)[0][0]
                ax.axvline(x=cutoff - 0.5, color=color, linestyle=':', alpha=0.5)
        ax.set_xlabel('Output position', fontsize=10)
        ax.set_title(f'Skip Layer {layer}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel('Conditional per-token accuracy', fontsize=10)
    fig.suptitle(f'Per-token accuracy (given prefix correct) with attention removed\n{tag}',
                 fontsize=11, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, 'ablation_conditional_accuracy.png'),
                dpi=300, bbox_inches='tight')
    plt.close()


def _plot_baseline(data, plot_dir, tag):
    fig, ax = plt.subplots(figsize=(4, 4))
    bw = 0.4
    vals = [data['LN0']['full_seq_acc'], data['LN1']['full_seq_acc']]
    bars = ax.bar([0, 1], vals, bw, color=['#6a3d9a', '#e6850e'])
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width()/2, h + 0.01,
                f'{h:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Without LayerNorm', 'With LayerNorm'], fontsize=11)
    ax.set_ylabel('Full-sequence accuracy', fontsize=12)
    ax.set_title(f'Baseline accuracy (500 trials)\n{tag}', fontsize=13, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.2, linestyle=':')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0, min(1.15, max(vals) * 1.2 + 0.05))
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, 'baseline_accuracy.png'), dpi=300, bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(6, 4))
    for ln, color, marker, label in [
        ('LN0', '#6a3d9a', 'o', 'Without LayerNorm'),
        ('LN1', '#e6850e', 's', 'With LayerNorm'),
    ]:
        ca = data[ln]['cond_acc']
        ce = data[ln]['cond_eligible']
        pos = np.arange(len(ca))
        valid = ce >= 10
        ax.plot(pos[valid], ca[valid], marker=marker, markersize=3,
                linewidth=1.2, label=label, color=color)
    ax.set_xlabel('Output position', fontsize=10)
    ax.set_ylabel('Conditional per-token accuracy', fontsize=10)
    ax.set_title(f'Per-token accuracy (given correct prefix) — baseline\n{tag}',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, 'baseline_conditional_accuracy.png'),
                dpi=300, bbox_inches='tight')
    plt.close()


def assemble_plots_for_iteration(itr):
    tmp_dir = os.path.join(OUTPUT_BASE, 'tmp_perckpt', f'{CONFIG_BASE}_itr{itr}')
    plot_dir = os.path.join(
        PARENT_PLOT_DIR,
        f'plots_{CONFIG_BASE}_{ARCH_SUFFIX}_ckpt{itr}'
    )
    os.makedirs(plot_dir, exist_ok=True)

    m = re.match(r'V(\d+)_B(\d+)_LR([\d.e+-]+)_MI(\d+)', CONFIG_BASE)
    tag = f'vocab={m.group(1)}  block={m.group(2)}  lr={m.group(3)}  iters={m.group(4)}  ckpt={itr}'

    def load(name):
        p = os.path.join(tmp_dir, name)
        if os.path.exists(p):
            return np.load(p)
        return None

    try:
        for layer in [0, 1]:
            r0 = load(f'cinclogits_LN0_layer{layer}.npz')
            r1 = load(f'cinclogits_LN1_layer{layer}.npz')
            if r0 is not None and r1 is not None:
                _plot_cinclogits(r0, r1, plot_dir, layer, tag)

            i0 = load(f'intensity_LN0_layer{layer}.npz')
            i1 = load(f'intensity_LN1_layer{layer}.npz')
            if i0 is not None and i1 is not None:
                _plot_intensity(i0, i1, layer, plot_dir, tag)

            i0_ub = load(f'intensity_LN0_layer{layer}_ub60.npz')
            i1_ub = load(f'intensity_LN1_layer{layer}_ub60.npz')
            if i0_ub is not None and i1_ub is not None:
                _plot_intensity(i0_ub, i1_ub, layer, plot_dir, tag, ub_suffix='60')

        abl_data = {}
        for ln in ['LN0', 'LN1']:
            for layer in [0, 1]:
                d = load(f'ablation_{ln}_layer{layer}.npz')
                if d is not None:
                    abl_data[(ln, layer)] = {
                        'full_seq_acc': float(d['full_seq_acc']),
                        'per_pos_acc': d['per_pos_acc'],
                        'cond_acc': d.get('cond_acc', None),
                        'cond_eligible': d.get('cond_eligible', None),
                    }
        if len(abl_data) == 4:
            _plot_ablation(abl_data, plot_dir, tag)

        bl_data = {}
        for ln in ['LN0', 'LN1']:
            d = load(f'baseline_{ln}.npz')
            if d is not None:
                bl_data[ln] = {
                    'full_seq_acc': float(d['full_seq_acc']),
                    'per_pos_acc': d['per_pos_acc'],
                    'cond_acc': d['cond_acc'],
                    'cond_eligible': d['cond_eligible'],
                }
        if len(bl_data) == 2:
            _plot_baseline(bl_data, plot_dir, tag)

        n_plots = len([f for f in os.listdir(plot_dir) if f.endswith('.png')])
        print(f'  ckpt{itr}: {n_plots} plots -> {plot_dir}')
    except Exception as e:
        print(f'  ERROR assembling ckpt{itr}: {e}')


def main():
    iterations = find_checkpoint_iterations()
    print(f'Config: {CONFIG_BASE}')
    print(f'Found {len(iterations)} checkpoint iterations: {iterations[0]}..{iterations[-1]}')
    print(f'Output: {PARENT_PLOT_DIR}')

    all_tasks = build_tasks(iterations)
    tasks_to_run = [t for t in all_tasks if not os.path.exists(t['out'])]
    print(f'Total tasks: {len(all_tasks)}, cached: {len(all_tasks) - len(tasks_to_run)}, '
          f'to run: {len(tasks_to_run)}')

    if tasks_to_run:
        print(f'\nDispatching {len(tasks_to_run)} workers across {NUM_GPUS} GPUs...')
        run_workers(tasks_to_run)

    print(f'\nAssembling plots for {len(iterations)} checkpoints...')
    for itr in iterations:
        assemble_plots_for_iteration(itr)

    print(f'\nDone! All plots in: {PARENT_PLOT_DIR}')


if __name__ == '__main__':
    main()
