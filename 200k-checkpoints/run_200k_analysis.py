"""
Parallel launcher for 200k-checkpoint analysis.
Generates all analysis tasks, distributes across 8 GPUs, assembles plots.
Produces plots compatible with grid-run style in subfolder outputs/plots_XXX/.
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
OUTPUT_BASE = os.path.join(SCRIPT_DIR, 'outputs')

NUM_GPUS = 8
UB_VALUES = [5, 10, 15, 20, 30, 50, 60]


def discover_checkpoints():
    pt_files = sorted(glob.glob(os.path.join(SCRIPT_DIR, '*.pt')))
    checkpoints = []
    for pt in pt_files:
        bn = os.path.basename(pt)
        parts = bn.replace('.pt', '').split('__')
        config_str = parts[0]
        ckpt_type = parts[1] if len(parts) > 1 else 'final'

        tokens = config_str.split('_')
        params = {}
        for t in tokens:
            if t.startswith('dseed'):
                params['dseed'] = t.replace('dseed', '')
            elif t.startswith('iseed'):
                params['iseed'] = t.replace('iseed', '')
            elif t.startswith('N'):
                try:
                    params['vocab'] = int(t[1:])
                except ValueError:
                    pass
            elif t.startswith('k'):
                try:
                    params['block'] = int(t[1:])
                except ValueError:
                    pass
            elif t.startswith('E'):
                try:
                    params['embd'] = int(t[1:])
                except ValueError:
                    pass
            elif t.startswith('L'):
                try:
                    params['layers'] = int(t[1:])
                except ValueError:
                    pass
            elif t.startswith('lr'):
                params['lr'] = t[2:].replace('p', '.')

        if ckpt_type.startswith('ckpt'):
            itr = int(ckpt_type.replace('ckpt', ''))
        else:
            itr = 200000

        vs = params.get('vocab', 256)
        bs = params.get('block', 16)
        lr = params.get('lr', '0.03')
        ds = params.get('dseed', '1337')
        iseed = params.get('iseed', '1337')

        lr_sci = f"{float(lr):.0e}".replace('+0', '+').replace('-0', '-')
        if lr_sci.endswith('+0'):
            lr_sci = lr_sci[:-2] + '+0'

        folder_name = f"plots_V{vs}_B{bs}_LR{lr_sci}_MI{itr}_E64_H1_L2_ds{ds}_is{iseed}_ckpt{itr}"

        checkpoints.append({
            'path': pt,
            'vocab': vs, 'block': bs, 'lr': lr, 'lr_sci': lr_sci,
            'dseed': ds, 'iseed': iseed, 'itr': itr,
            'folder_name': folder_name,
        })
    return checkpoints


def make_tasks(ckpt):
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


def assemble_plots(ckpt):
    folder_name = ckpt['folder_name']
    tmp_dir = os.path.join(OUTPUT_BASE, 'tmp_results', folder_name)
    plot_dir = os.path.join(OUTPUT_BASE, folder_name)
    os.makedirs(plot_dir, exist_ok=True)
    tag = (f"V={ckpt['vocab']}  B={ckpt['block']}  lr={ckpt['lr']}  "
           f"iters={ckpt['itr']}  dseed={ckpt['dseed']}  iseed={ckpt['iseed']}")

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
    ax.set_xticklabels(['Model (with LN)'], fontsize=12)
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
            plt.xticks(intensities[::2], fontsize=8)
            plt.yticks(fontsize=8)
            plt.ylim(0, 1.05)
            plt.tight_layout()
            fname = f'intensity_layer{layer}'
            if ub != 5:
                fname += f'_ub{ub}'
            plt.savefig(os.path.join(plot_dir, f'{fname}.png'),
                        dpi=300, bbox_inches='tight')
            plt.close()


def main():
    t_start = time.time()

    checkpoints = discover_checkpoints()
    print(f"Found {len(checkpoints)} checkpoints")
    for ckpt in checkpoints:
        print(f"  {os.path.basename(ckpt['path'])} -> {ckpt['folder_name']}")

    all_tasks = []
    for ckpt in checkpoints:
        all_tasks.extend(make_tasks(ckpt))

    cached = sum(1 for t in all_tasks if os.path.exists(t['out']))
    to_run = [t for t in all_tasks if not os.path.exists(t['out'])]
    print(f"\nTotal tasks: {len(all_tasks)}, cached: {cached}, to run: {len(to_run)}")

    if not to_run:
        print("All tasks cached. Assembling plots only...")
        for ckpt in checkpoints:
            n = assemble_plots(ckpt)
            print(f"  {ckpt['folder_name']}: {n} plots")
        print(f"\nDone in {time.time() - t_start:.0f}s")
        return

    gpu_tasks = {g: [] for g in range(NUM_GPUS)}
    for i, task in enumerate(to_run):
        gpu_tasks[i % NUM_GPUS].append(task)

    print(f"\nDistributing {len(to_run)} tasks across {NUM_GPUS} GPUs:")
    for g in range(NUM_GPUS):
        print(f"  GPU {g}: {len(gpu_tasks[g])} tasks")

    task_dir = os.path.join(OUTPUT_BASE, 'task_files')
    os.makedirs(task_dir, exist_ok=True)
    log_dir = os.path.join(OUTPUT_BASE, 'gpu_worker_logs')
    os.makedirs(log_dir, exist_ok=True)

    procs = {}
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
        procs[g] = (proc, log_file)

    print(f"\nLaunched {len(procs)} GPU workers. Monitoring...\n", flush=True)

    while any(p.poll() is None for p, _ in procs.values()):
        time.sleep(5)
        done_now = sum(1 for t in all_tasks if os.path.exists(t['out']))
        elapsed = time.time() - t_start
        print(f"  [PROGRESS] {done_now}/{len(all_tasks)} tasks done ({elapsed:.0f}s)", flush=True)

    for g, (proc, lf) in procs.items():
        lf.close()
        if proc.returncode != 0:
            print(f"  [WARN] GPU {g} worker exited with code {proc.returncode}", flush=True)

    print("\nAssembling plots...", flush=True)
    for ckpt in checkpoints:
        n = assemble_plots(ckpt)
        print(f"  {ckpt['folder_name']}: {n} plots")

    total_elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"ALL DONE — {len(checkpoints)} checkpoints analyzed and plotted")
    print(f"Elapsed: {total_elapsed:.0f}s ({total_elapsed/60:.1f}m)")
    print(f"Output: {OUTPUT_BASE}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
