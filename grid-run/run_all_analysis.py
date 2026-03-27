"""
Parallel launcher for analysis. Dispatches fine-grained worker processes
(one per checkpoint × task × layer) across GPUs, then assembles plots.

Each worker is fully independent — no shared model state.
"""
import argparse
import itertools
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

VOCAB_SIZES = [64, 128, 256, 512, 8192]
BLOCK_SIZES = [16, 32]
MAX_ITERS = [10000, 20000, 40000, 60000]
LEARNING_RATES = ['1e-04', '1e-03', '1e-02']

NUM_GPUS = 8
GPU_MEM_TOTAL = 81559
MEM_CAP_FRAC = 0.90
MAX_WORKERS_PER_GPU = 10

WORKER_MEM_EST = {64: 200, 128: 300, 256: 400, 512: 600, 8192: 2000}


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
    except:
        return {i: 0 for i in range(NUM_GPUS)}


def find_final_ckpt(run_dir, max_iters):
    if not os.path.isdir(run_dir):
        return None
    target = f"_itr{max_iters}.pt"
    for f in os.listdir(run_dir):
        if f.endswith(target):
            return os.path.join(run_dir, f)
    return None


def find_ready_pairs():
    pairs = []
    for vs, bs, lr, mi in itertools.product(VOCAB_SIZES, BLOCK_SIZES, LEARNING_RATES, MAX_ITERS):
        dir_0 = os.path.join(OUTPUT_BASE, f"V{vs}_B{bs}_LR{lr}_MI{mi}_LN0_E64_H1_L2")
        dir_1 = os.path.join(OUTPUT_BASE, f"V{vs}_B{bs}_LR{lr}_MI{mi}_LN1_E64_H1_L2")
        ckpt_0 = find_final_ckpt(dir_0, mi)
        ckpt_1 = find_final_ckpt(dir_1, mi)
        if ckpt_0 and ckpt_1:
            pairs.append({
                'vocab_size': vs, 'block_size': bs, 'lr': lr, 'max_iters': mi,
                'ckpt_0': ckpt_0, 'ckpt_1': ckpt_1,
            })
    return pairs


def is_analysis_done(pair):
    vs, bs, lr, mi = pair['vocab_size'], pair['block_size'], pair['lr'], pair['max_iters']
    prefix = f"plots_V{vs}_B{bs}_LR{lr}_MI{mi}_E64_H1_L2_ckpt"
    expected = [
        'compare_cinclogits_layer0.png', 'compare_cinclogits_layer1.png',
        'compare_intensity_layer0.png', 'compare_intensity_layer1.png',
    ]
    if not os.path.isdir(OUTPUT_BASE):
        return False
    for d in os.listdir(OUTPUT_BASE):
        if d.startswith(prefix) and os.path.isdir(os.path.join(OUTPUT_BASE, d)):
            plot_dir = os.path.join(OUTPUT_BASE, d)
            if all(os.path.exists(os.path.join(plot_dir, f)) for f in expected):
                return True
    return False


def make_tasks(pairs, ub_values=None):
    """Generate all worker tasks. Each task = one process.
    If ub_values is set, generate only intensity tasks for those ub values."""
    tasks = []
    for pair in pairs:
        vs, bs, lr, mi = pair['vocab_size'], pair['block_size'], pair['lr'], pair['max_iters']
        base = f"V{vs}_B{bs}_LR{lr}_MI{mi}"
        tmp_dir = os.path.join(OUTPUT_BASE, 'tmp_results', base)
        for ln, ckpt in [('LN0', pair['ckpt_0']), ('LN1', pair['ckpt_1'])]:
            if ub_values is not None:
                for ub in ub_values:
                    for layer in [0, 1]:
                        out_file = os.path.join(tmp_dir, f"intensity_{ln}_layer{layer}_ub{ub}.npz")
                        tasks.append({
                            'pair': pair, 'ln': ln, 'ckpt': ckpt,
                            'task': 'intensity', 'layer': layer,
                            'unsorted_lb': ub, 'unsorted_ub': ub,
                            'out': out_file,
                            'name': f"{base}_intensity_ub{ub}_{ln}_L{layer}",
                            'est_mem': WORKER_MEM_EST.get(vs, 500),
                        })
            else:
                for task_type in ['cinclogits', 'intensity']:
                    for layer in [0, 1]:
                        out_file = os.path.join(tmp_dir, f"{task_type}_{ln}_layer{layer}.npz")
                        tasks.append({
                            'pair': pair, 'ln': ln, 'ckpt': ckpt,
                            'task': task_type, 'layer': layer,
                            'out': out_file,
                            'name': f"{base}_{task_type}_{ln}_L{layer}",
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


def assemble_plots(pair):
    """Read cached results and produce the 4 plots for a config pair."""
    vs, bs, lr, mi = pair['vocab_size'], pair['block_size'], pair['lr'], pair['max_iters']
    base = f"V{vs}_B{bs}_LR{lr}_MI{mi}"
    tmp_dir = os.path.join(OUTPUT_BASE, 'tmp_results', base)

    def load_result(task_type, ln, layer):
        path = os.path.join(tmp_dir, f"{task_type}_{ln}_layer{layer}.npz")
        return np.load(path)

    itr_0 = int(load_result('cinclogits', 'LN0', 0)['itr'])
    itr_1 = int(load_result('cinclogits', 'LN1', 0)['itr'])
    ckpt_itr = itr_0 if itr_0 == itr_1 else max(itr_0, itr_1)
    ckpt_str = f"ckpt={itr_0}" if itr_0 == itr_1 else f"ckpt LN0={itr_0} LN1={itr_1}"
    tag = f"vocab={vs}  block={bs}  lr={lr}  iters={mi}  {ckpt_str}"

    plot_dir = os.path.join(OUTPUT_BASE, f"plots_{base}_E64_H1_L2_ckpt{ckpt_itr}")
    os.makedirs(plot_dir, exist_ok=True)

    for layer in [0, 1]:
        # cinclogits plot
        r0 = load_result('cinclogits', 'LN0', layer)
        r1 = load_result('cinclogits', 'LN1', layer)
        _plot_cinclogits(
            (r0['clogit_icscore'], r0['iclogit_icscore']),
            (r1['clogit_icscore'], r1['iclogit_icscore']),
            plot_dir, attn_layer=layer, tag=tag)

        # intensity plot
        i0 = load_result('intensity', 'LN0', layer)
        i1 = load_result('intensity', 'LN1', layer)
        _plot_intensity(
            (i0['intensities'], i0['success_rates']),
            (i1['intensities'], i1['success_rates']),
            attn_layer=layer, plot_dir=plot_dir, tag=tag)

    print(f"  Plots saved to {plot_dir}")


def _plot_cinclogits(results_ln0, results_ln1, plot_dir, attn_layer=0, tag=""):
    cl_ic_0, icl_ic_0 = results_ln0
    cl_ic_1, icl_ic_1 = results_ln1
    frac_ic_0 = np.mean(cl_ic_0 + icl_ic_0)
    frac_ic_1 = np.mean(cl_ic_1 + icl_ic_1)
    eps = 1e-10
    corr_0 = np.sum(cl_ic_0) / (np.sum(cl_ic_0 + icl_ic_0) + eps)
    corr_1 = np.sum(cl_ic_1) / (np.sum(cl_ic_1 + icl_ic_1) + eps)

    fig, ax = plt.subplots(figsize=(5, 4.2))
    bw = 0.32
    x = np.array([0, 1])
    b1 = ax.bar(x[0] - bw / 2, frac_ic_0, bw, color='#6a3d9a', label='Without LayerNorm')
    b2 = ax.bar(x[0] + bw / 2, frac_ic_1, bw, color='#e6850e', label='With LayerNorm')
    b3 = ax.bar(x[1] - bw / 2, corr_0, bw, color='#6a3d9a')
    b4 = ax.bar(x[1] + bw / 2, corr_1, bw, color='#e6850e')
    for bar in [b1, b2, b3, b4]:
        for b in bar:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, h + 0.008,
                    f'{h:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Fraction of\nincorrect scores',
                        'Logit correction ratio\namong incorrect scores'], fontsize=11)
    ax.set_ylabel('Fraction', fontsize=12)
    title = f'Incorrect scores & logit correction (Layer {attn_layer})'
    if tag:
        title += f'\n{tag}'
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(True, axis='y', alpha=0.2, linestyle=':')
    ax.tick_params(labelsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ymax = max(frac_ic_0, frac_ic_1, corr_0, corr_1)
    ax.set_ylim(0, ymax * 1.25 if ymax > 0 else 1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, f'compare_cinclogits_layer{attn_layer}.png'),
                dpi=300, bbox_inches='tight')
    plt.close()


def _plot_intensity(results_ln0, results_ln1, attn_layer, plot_dir, tag="", ub_suffix=""):
    intensities_0, rates_0 = results_ln0
    intensities_1, rates_1 = results_ln1

    plt.figure(figsize=(3.5, 2.8))
    plt.plot(intensities_0, rates_0, marker='o', linewidth=1.5, markersize=5,
             label='without layer norm', color='#1f77b4')
    plt.plot(intensities_1, rates_1, marker='s', linewidth=1.5, markersize=5,
             label='with layer norm', color='#ff7f0e')
    plt.xlabel('Intervention Intensity', fontsize=9)
    plt.ylabel('Success Probability', fontsize=9)
    title = f'Robustness to Attention Intervention (Layer {attn_layer})'
    if ub_suffix:
        title += f'  [ub={ub_suffix}]'
    if tag:
        title += f'\n{tag}'
    plt.title(title, fontsize=10)
    plt.legend(fontsize=7, loc='lower left')
    plt.grid(True, alpha=0.3)
    plt.xticks(list(intensities_0[::2]), fontsize=8)
    plt.yticks(fontsize=8)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    fname = f'compare_intensity_layer{attn_layer}'
    if ub_suffix:
        fname += f'_ub{ub_suffix}'
    plt.savefig(os.path.join(plot_dir, f'{fname}.png'),
                dpi=300, bbox_inches='tight')
    plt.close()


# ── UB-variant assembly ───────────────────────────────────────────────────────

def assemble_ub_plots(pair, ub_values):
    """Produce intensity plots for non-default ub values into existing plot dirs."""
    vs, bs, lr, mi = pair['vocab_size'], pair['block_size'], pair['lr'], pair['max_iters']
    base = f"V{vs}_B{bs}_LR{lr}_MI{mi}"
    tmp_dir = os.path.join(OUTPUT_BASE, 'tmp_results', base)

    itr_0 = int(np.load(os.path.join(tmp_dir, 'cinclogits_LN0_layer0.npz'))['itr'])
    itr_1 = int(np.load(os.path.join(tmp_dir, 'cinclogits_LN1_layer0.npz'))['itr'])
    ckpt_itr = itr_0 if itr_0 == itr_1 else max(itr_0, itr_1)
    ckpt_str = f"ckpt={itr_0}" if itr_0 == itr_1 else f"ckpt LN0={itr_0} LN1={itr_1}"
    tag = f"vocab={vs}  block={bs}  lr={lr}  iters={mi}  {ckpt_str}"

    plot_dir = os.path.join(OUTPUT_BASE, f"plots_{base}_E64_H1_L2_ckpt{ckpt_itr}")
    os.makedirs(plot_dir, exist_ok=True)

    for ub in ub_values:
        for layer in [0, 1]:
            f0 = os.path.join(tmp_dir, f"intensity_LN0_layer{layer}_ub{ub}.npz")
            f1 = os.path.join(tmp_dir, f"intensity_LN1_layer{layer}_ub{ub}.npz")
            if not os.path.exists(f0) or not os.path.exists(f1):
                continue
            i0 = np.load(f0)
            i1 = np.load(f1)
            _plot_intensity(
                (i0['intensities'], i0['success_rates']),
                (i1['intensities'], i1['success_rates']),
                attn_layer=layer, plot_dir=plot_dir, tag=tag, ub_suffix=str(ub))
    print(f"  UB plots saved to {plot_dir}")


# ── Ablation assembly ─────────────────────────────────────────────────────────

def assemble_ablation_plots(pair):
    """Produce ablation accuracy plot for a config pair."""
    vs, bs, lr, mi = pair['vocab_size'], pair['block_size'], pair['lr'], pair['max_iters']
    base = f"V{vs}_B{bs}_LR{lr}_MI{mi}"
    tmp_dir = os.path.join(OUTPUT_BASE, 'tmp_results', base)

    itr_0 = int(np.load(os.path.join(tmp_dir, 'cinclogits_LN0_layer0.npz'))['itr'])
    itr_1 = int(np.load(os.path.join(tmp_dir, 'cinclogits_LN1_layer0.npz'))['itr'])
    ckpt_itr = itr_0 if itr_0 == itr_1 else max(itr_0, itr_1)
    ckpt_str = f"ckpt={itr_0}" if itr_0 == itr_1 else f"ckpt LN0={itr_0} LN1={itr_1}"
    tag = f"vocab={vs}  block={bs}  lr={lr}  iters={mi}  {ckpt_str}"

    plot_dir = os.path.join(OUTPUT_BASE, f"plots_{base}_E64_H1_L2_ckpt{ckpt_itr}")
    os.makedirs(plot_dir, exist_ok=True)

    data = {}
    for ln in ['LN0', 'LN1']:
        for layer in [0, 1]:
            f = os.path.join(tmp_dir, f"ablation_{ln}_layer{layer}.npz")
            if not os.path.exists(f):
                return
            d = np.load(f)
            entry = {
                'full_seq_acc': float(d['full_seq_acc']),
                'per_pos_acc': d['per_pos_acc'],
            }
            if 'cond_acc' in d:
                entry['cond_acc'] = d['cond_acc']
                entry['cond_eligible'] = d['cond_eligible']
            data[(ln, layer)] = entry

    _plot_ablation(data, plot_dir, tag)
    print(f"  Ablation plot saved to {plot_dir}")


def _plot_ablation(data, plot_dir, tag=""):
    fig, ax = plt.subplots(figsize=(6, 4.5))

    labels = ['Skip Layer 0', 'Skip Layer 1']
    x = np.arange(len(labels))
    bw = 0.3

    ln0_vals = [data[('LN0', 0)]['full_seq_acc'], data[('LN0', 1)]['full_seq_acc']]
    ln1_vals = [data[('LN1', 0)]['full_seq_acc'], data[('LN1', 1)]['full_seq_acc']]

    b1 = ax.bar(x - bw / 2, ln0_vals, bw, color='#6a3d9a', label='Without LayerNorm')
    b2 = ax.bar(x + bw / 2, ln1_vals, bw, color='#e6850e', label='With LayerNorm')

    for bars in [b1, b2]:
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
    ax.set_ylim(0, min(1.15, max(ln0_vals + ln1_vals) * 1.25 + 0.05))
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, 'ablation_accuracy.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # Per-position accuracy plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for i, layer in enumerate([0, 1]):
        ax = axes[i]
        pos = np.arange(len(data[('LN0', layer)]['per_pos_acc']))
        ax.plot(pos, data[('LN0', layer)]['per_pos_acc'],
                marker='o', markersize=3, linewidth=1.2,
                label='Without LayerNorm', color='#6a3d9a')
        ax.plot(pos, data[('LN1', layer)]['per_pos_acc'],
                marker='s', markersize=3, linewidth=1.2,
                label='With LayerNorm', color='#e6850e')
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

    if 'cond_acc' not in data[('LN0', 0)]:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for i, layer in enumerate([0, 1]):
        ax = axes[i]
        for ln, color, marker, label in [
            ('LN0', '#6a3d9a', 'o', 'Without LayerNorm'),
            ('LN1', '#e6850e', 's', 'With LayerNorm'),
        ]:
            cond_acc = data[(ln, layer)]['cond_acc']
            cond_elig = data[(ln, layer)]['cond_eligible']
            pos = np.arange(len(cond_acc))
            valid = cond_elig >= 10
            ax.plot(pos[valid], cond_acc[valid],
                    marker=marker, markersize=3, linewidth=1.2,
                    label=label, color=color)
            if not valid.all():
                cutoff = np.where(~valid)[0][0]
                ax.axvline(x=cutoff - 0.5, color=color, linestyle=':', alpha=0.5)
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


# ── Baseline assembly ─────────────────────────────────────────────────────────

def assemble_baseline_plots(pair):
    """Produce baseline accuracy plots for a config pair."""
    vs, bs, lr, mi = pair['vocab_size'], pair['block_size'], pair['lr'], pair['max_iters']
    base = f"V{vs}_B{bs}_LR{lr}_MI{mi}"
    tmp_dir = os.path.join(OUTPUT_BASE, 'tmp_results', base)

    itr_0 = int(np.load(os.path.join(tmp_dir, 'cinclogits_LN0_layer0.npz'))['itr'])
    itr_1 = int(np.load(os.path.join(tmp_dir, 'cinclogits_LN1_layer0.npz'))['itr'])
    ckpt_itr = itr_0 if itr_0 == itr_1 else max(itr_0, itr_1)
    ckpt_str = f"ckpt={itr_0}" if itr_0 == itr_1 else f"ckpt LN0={itr_0} LN1={itr_1}"
    tag = f"vocab={vs}  block={bs}  lr={lr}  iters={mi}  {ckpt_str}"

    plot_dir = os.path.join(OUTPUT_BASE, f"plots_{base}_E64_H1_L2_ckpt{ckpt_itr}")
    os.makedirs(plot_dir, exist_ok=True)

    data = {}
    for ln in ['LN0', 'LN1']:
        f = os.path.join(tmp_dir, f"baseline_{ln}.npz")
        if not os.path.exists(f):
            return
        d = np.load(f)
        data[ln] = {
            'full_seq_acc': float(d['full_seq_acc']),
            'per_pos_acc': d['per_pos_acc'],
            'cond_acc': d['cond_acc'],
            'cond_eligible': d['cond_eligible'],
        }

    # Full-sequence accuracy bar chart
    fig, ax = plt.subplots(figsize=(4, 4))
    bw = 0.4
    vals = [data['LN0']['full_seq_acc'], data['LN1']['full_seq_acc']]
    bars = ax.bar([0, 1], vals, bw, color=['#6a3d9a', '#e6850e'])
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + 0.01,
                f'{h:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Without LayerNorm', 'With LayerNorm'], fontsize=11)
    ax.set_ylabel('Full-sequence accuracy', fontsize=12)
    title = 'Baseline accuracy (500 trials)'
    if tag:
        title += f'\n{tag}'
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.2, linestyle=':')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0, min(1.15, max(vals) * 1.2 + 0.05))
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, 'baseline_accuracy.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # Conditional per-token accuracy
    fig, ax = plt.subplots(figsize=(6, 4))
    for ln, color, marker, label in [
        ('LN0', '#6a3d9a', 'o', 'Without LayerNorm'),
        ('LN1', '#e6850e', 's', 'With LayerNorm'),
    ]:
        cond_acc = data[ln]['cond_acc']
        cond_elig = data[ln]['cond_eligible']
        pos = np.arange(len(cond_acc))
        valid = cond_elig >= 10
        ax.plot(pos[valid], cond_acc[valid],
                marker=marker, markersize=3, linewidth=1.2,
                label=label, color=color)
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

    print(f"  Baseline plots saved to {plot_dir}")


# ── Worker dispatch loop ──────────────────────────────────────────────────────

def run_workers(tasks_to_run, queue):
    gpu_jobs = {g: [] for g in range(NUM_GPUS)}
    gpu_analysis_mem = {g: 0 for g in range(NUM_GPUS)}
    completed = 0
    failed = 0
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
                        if completed % 20 == 0:
                            elapsed = time.time() - t0
                            print(f"[PROGRESS] {completed}/{len(tasks_to_run)} done, {failed} fail, {len(queue)} queued ({elapsed:.0f}s)")
                    else:
                        failed += 1
                        print(f"[FAIL] GPU {g}: {name} exit={ret}")
                else:
                    still_running.append((proc, log_file, name, mem))
            gpu_jobs[g] = still_running

        mem_used = get_gpu_mem_used()

        remaining = []
        for task in queue:
            placed = False
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
                placed = True
            if not placed:
                remaining.append(task)
        queue = remaining

        if any(gpu_jobs[g] for g in range(NUM_GPUS)):
            time.sleep(3)

    elapsed = time.time() - t0
    print(f"\nAll workers done! completed={completed}, failed={failed}, elapsed={elapsed:.0f}s")


# ── Main loop ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--intensity-ub', type=int, nargs='+', default=None,
                        help='Run intensity tasks for these ub values only (e.g. --intensity-ub 10 15)')
    parser.add_argument('--ablation', action='store_true',
                        help='Run ablation tasks (skip attention+layernorm per layer)')
    parser.add_argument('--baseline', action='store_true',
                        help='Run baseline accuracy tasks (intact model)')
    args = parser.parse_args()

    pairs = find_ready_pairs()
    print(f"Found {len(pairs)} completed config pairs")

    if args.ablation:
        print("Mode: ablation (skip attention+LN per layer)")
        all_tasks = []
        for pair in pairs:
            vs, bs, lr, mi = pair['vocab_size'], pair['block_size'], pair['lr'], pair['max_iters']
            base = f"V{vs}_B{bs}_LR{lr}_MI{mi}"
            tmp_dir = os.path.join(OUTPUT_BASE, 'tmp_results', base)
            for ln, ckpt in [('LN0', pair['ckpt_0']), ('LN1', pair['ckpt_1'])]:
                for layer in [0, 1]:
                    out_file = os.path.join(tmp_dir, f"ablation_{ln}_layer{layer}.npz")
                    all_tasks.append({
                        'pair': pair, 'ln': ln, 'ckpt': ckpt,
                        'task': 'ablation', 'layer': layer,
                        'out': out_file,
                        'name': f"{base}_ablation_{ln}_L{layer}",
                        'est_mem': WORKER_MEM_EST.get(vs, 500),
                    })
        tasks_to_run = [t for t in all_tasks if not is_task_done(t)]
        print(f"Total tasks: {len(all_tasks)}, cached: {len(all_tasks) - len(tasks_to_run)}, to run: {len(tasks_to_run)}")

        if args.dry_run:
            for t in tasks_to_run[:20]:
                print(f"  {t['name']}")
            if len(tasks_to_run) > 20:
                print(f"  ... and {len(tasks_to_run) - 20} more")
            return

        tasks_to_run.sort(key=lambda t: t['pair']['vocab_size'])
        run_workers(tasks_to_run, list(tasks_to_run))

        print("\nAssembling ablation plots...")
        for pair in pairs:
            vs, bs, lr, mi = pair['vocab_size'], pair['block_size'], pair['lr'], pair['max_iters']
            base = f"V{vs}_B{bs}_LR{lr}_MI{mi}"
            try:
                assemble_ablation_plots(pair)
            except Exception as e:
                print(f"  ERROR assembling ablation for {base}: {e}")

        print("\nDone!")
        return

    if args.baseline:
        print("Mode: baseline accuracy (intact model)")
        all_tasks = []
        for pair in pairs:
            vs, bs, lr, mi = pair['vocab_size'], pair['block_size'], pair['lr'], pair['max_iters']
            base = f"V{vs}_B{bs}_LR{lr}_MI{mi}"
            tmp_dir = os.path.join(OUTPUT_BASE, 'tmp_results', base)
            for ln, ckpt in [('LN0', pair['ckpt_0']), ('LN1', pair['ckpt_1'])]:
                out_file = os.path.join(tmp_dir, f"baseline_{ln}.npz")
                all_tasks.append({
                    'pair': pair, 'ln': ln, 'ckpt': ckpt,
                    'task': 'baseline', 'layer': 0,
                    'out': out_file,
                    'name': f"{base}_baseline_{ln}",
                    'est_mem': WORKER_MEM_EST.get(vs, 500),
                })
        tasks_to_run = [t for t in all_tasks if not is_task_done(t)]
        print(f"Total tasks: {len(all_tasks)}, cached: {len(all_tasks) - len(tasks_to_run)}, to run: {len(tasks_to_run)}")

        if args.dry_run:
            for t in tasks_to_run[:20]:
                print(f"  {t['name']}")
            if len(tasks_to_run) > 20:
                print(f"  ... and {len(tasks_to_run) - 20} more")
            return

        tasks_to_run.sort(key=lambda t: t['pair']['vocab_size'])
        run_workers(tasks_to_run, list(tasks_to_run))

        print("\nAssembling baseline plots...")
        for pair in pairs:
            vs, bs, lr, mi = pair['vocab_size'], pair['block_size'], pair['lr'], pair['max_iters']
            base = f"V{vs}_B{bs}_LR{lr}_MI{mi}"
            try:
                assemble_baseline_plots(pair)
            except Exception as e:
                print(f"  ERROR assembling baseline for {base}: {e}")

        print("\nDone!")
        return

    if args.intensity_ub:
        ub_values = args.intensity_ub
        print(f"Mode: intensity-only for ub={ub_values}")
        all_tasks = make_tasks(pairs, ub_values=ub_values)
        tasks_to_run = [t for t in all_tasks if not is_task_done(t)]
        print(f"Total tasks: {len(all_tasks)}, cached: {len(all_tasks) - len(tasks_to_run)}, to run: {len(tasks_to_run)}")

        if args.dry_run:
            for t in tasks_to_run[:20]:
                print(f"  {t['name']}")
            if len(tasks_to_run) > 20:
                print(f"  ... and {len(tasks_to_run) - 20} more")
            return

        tasks_to_run.sort(key=lambda t: t['pair']['vocab_size'])
        run_workers(tasks_to_run, list(tasks_to_run))

        print("\nAssembling UB plots...")
        for pair in pairs:
            vs, bs, lr, mi = pair['vocab_size'], pair['block_size'], pair['lr'], pair['max_iters']
            base = f"V{vs}_B{bs}_LR{lr}_MI{mi}"
            try:
                assemble_ub_plots(pair, ub_values)
            except Exception as e:
                print(f"  ERROR assembling UB plots for {base}: {e}")

        print("\nDone!")
        return

    todo_pairs = [p for p in pairs if not is_analysis_done(p)]
    print(f"Already fully analyzed: {len(pairs) - len(todo_pairs)}, remaining: {len(todo_pairs)}")

    all_tasks = make_tasks(todo_pairs)
    tasks_to_run = [t for t in all_tasks if not is_task_done(t)]
    print(f"Total worker tasks: {len(all_tasks)}, already cached: {len(all_tasks) - len(tasks_to_run)}, to run: {len(tasks_to_run)}")

    if args.dry_run:
        for t in tasks_to_run[:20]:
            print(f"  {t['name']}")
        if len(tasks_to_run) > 20:
            print(f"  ... and {len(tasks_to_run) - 20} more")
        return

    tasks_to_run.sort(key=lambda t: (0 if t['task'] == 'cinclogits' else 1, t['pair']['vocab_size']))
    run_workers(tasks_to_run, list(tasks_to_run))

    print("\nAssembling plots...")
    for pair in todo_pairs:
        vs, bs, lr, mi = pair['vocab_size'], pair['block_size'], pair['lr'], pair['max_iters']
        base = f"V{vs}_B{bs}_LR{lr}_MI{mi}"
        tmp_dir = os.path.join(OUTPUT_BASE, 'tmp_results', base)
        expected_files = [f"{tt}_{ln}_layer{ly}.npz"
                          for tt in ['cinclogits', 'intensity']
                          for ln in ['LN0', 'LN1']
                          for ly in [0, 1]]
        if all(os.path.exists(os.path.join(tmp_dir, f)) for f in expected_files):
            try:
                assemble_plots(pair)
            except Exception as e:
                print(f"  ERROR assembling {base}: {e}")
        else:
            missing = [f for f in expected_files if not os.path.exists(os.path.join(tmp_dir, f))]
            print(f"  SKIP {base}: missing {len(missing)} results")

    print("\nDone!")


if __name__ == '__main__':
    main()
