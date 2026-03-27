"""
Unified analysis launcher for 1000k-checkpoints.
Produces the same 27 plots per checkpoint as 200k-checkpoints:
  - baseline_accuracy, baseline_conditional_accuracy
  - ablation_accuracy, ablation_per_position, ablation_conditional_accuracy
  - cinclogits_layer0, cinclogits_layer1
  - intensity_layer{0,1} (ub=5,10,15,20,30,50,60) = 14 plots
  - intensity_layer0_asym_ub60_lb60
  - hijack_{breaking_rate,hijack_rate,sample_count}_heatmap_layer0
  - intervention_pernumber_{separator,random}_layer0
Distributed across 8 GPUs with incremental plot assembly.
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
from matplotlib.colors import Normalize

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_BASE = os.path.join(SCRIPT_DIR, 'outputs')
NUM_GPUS = 8
UB_VALUES = [5, 10, 15, 20, 30, 50, 60]
INTENSITIES_SEP = [2.0, 6.0, 10.0]
BIN_SIZE = 8
N_BINS = 256 // BIN_SIZE


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
                try: params['vocab'] = int(t[1:])
                except ValueError: pass
            elif t.startswith('k'):
                try: params['block'] = int(t[1:])
                except ValueError: pass
            elif t.startswith('E'):
                try: params['embd'] = int(t[1:])
                except ValueError: pass
            elif t.startswith('L'):
                try: params['layers'] = int(t[1:])
                except ValueError: pass
            elif t.startswith('lr'):
                params['lr'] = t[2:].replace('p', '.')

        if ckpt_type.startswith('ckpt'):
            itr = int(ckpt_type.replace('ckpt', ''))
        elif ckpt_type == 'final':
            itr = 1000000
        else:
            itr = 0

        vs = params.get('vocab', 256)
        bs = params.get('block', 16)
        lr = params.get('lr', '0.03')
        ds = params.get('dseed', '1337')
        iseed = params.get('iseed', '1337')
        lr_sci = f"{float(lr):.0e}".replace('+0', '+').replace('-0', '-')

        folder_name = f"plots_V{vs}_B{bs}_LR{lr_sci}_MI{itr}_E64_H1_L2_ds{ds}_is{iseed}_ckpt{itr}"
        checkpoints.append({
            'path': pt, 'vocab': vs, 'block': bs, 'lr': lr, 'lr_sci': lr_sci,
            'dseed': ds, 'iseed': iseed, 'itr': itr, 'folder_name': folder_name,
        })

    seen = {}
    for ckpt in checkpoints:
        fn = ckpt['folder_name']
        if fn not in seen or 'final' in os.path.basename(ckpt['path']):
            seen[fn] = ckpt
    return sorted(seen.values(), key=lambda c: c['itr'])


def make_tasks(ckpt):
    tmp = os.path.join(OUTPUT_BASE, 'tmp_results', ckpt['folder_name'])
    tasks = []

    tasks.append({'ckpt_path': ckpt['path'], 'type': 'baseline', 'layer': 0,
                  'out': os.path.join(tmp, 'baseline.npz'),
                  'name': f"{ckpt['folder_name']}_baseline", 'itr': ckpt['itr']})

    for layer in [0, 1]:
        tasks.append({'ckpt_path': ckpt['path'], 'type': 'ablation', 'layer': layer,
                      'out': os.path.join(tmp, f'ablation_layer{layer}.npz'),
                      'name': f"{ckpt['folder_name']}_ablation_L{layer}", 'itr': ckpt['itr']})

    for layer in [0, 1]:
        tasks.append({'ckpt_path': ckpt['path'], 'type': 'cinclogits', 'layer': layer,
                      'out': os.path.join(tmp, f'cinclogits_layer{layer}.npz'),
                      'name': f"{ckpt['folder_name']}_cinclogits_L{layer}", 'itr': ckpt['itr']})

    for ub in UB_VALUES:
        for layer in [0, 1]:
            suffix = '' if ub == 5 else f'_ub{ub}'
            tasks.append({'ckpt_path': ckpt['path'], 'type': 'intensity', 'layer': layer,
                          'ub': ub,
                          'out': os.path.join(tmp, f'intensity_layer{layer}{suffix}.npz'),
                          'name': f"{ckpt['folder_name']}_intensity_ub{ub}_L{layer}",
                          'itr': ckpt['itr']})

    tasks.append({'ckpt_path': ckpt['path'], 'type': 'intensity_asym', 'layer': 0,
                  'unsorted_ub': 60, 'unsorted_lb': 0, 'unsorted_ub_num': 1, 'unsorted_lb_num': 0,
                  'out': os.path.join(tmp, 'intensity_layer0_ub60_lb0.npz'),
                  'name': f"{ckpt['folder_name']}_asym_ub60_lb0", 'itr': ckpt['itr']})
    tasks.append({'ckpt_path': ckpt['path'], 'type': 'intensity_asym', 'layer': 0,
                  'unsorted_ub': 0, 'unsorted_lb': 60, 'unsorted_ub_num': 0, 'unsorted_lb_num': 1,
                  'out': os.path.join(tmp, 'intensity_layer0_ub0_lb60.npz'),
                  'name': f"{ckpt['folder_name']}_asym_ub0_lb60", 'itr': ckpt['itr']})

    tasks.append({'ckpt_path': ckpt['path'], 'type': 'hijack', 'layer': 0,
                  'trials': 2000,
                  'out': os.path.join(tmp, 'hijack.npz'),
                  'name': f"{ckpt['folder_name']}_hijack", 'itr': ckpt['itr']})

    tasks.append({'ckpt_path': ckpt['path'], 'type': 'separator_random', 'layer': 0,
                  'trials': 1000,
                  'out': os.path.join(tmp, 'separator_random.npz'),
                  'name': f"{ckpt['folder_name']}_sep_rand", 'itr': ckpt['itr']})

    return tasks


def is_ckpt_done(ckpt):
    tmp = os.path.join(OUTPUT_BASE, 'tmp_results', ckpt['folder_name'])
    required = ['baseline.npz', 'hijack.npz', 'separator_random.npz',
                'intensity_layer0_ub60_lb0.npz', 'intensity_layer0_ub0_lb60.npz']
    for layer in [0, 1]:
        required.append(f'ablation_layer{layer}.npz')
        required.append(f'cinclogits_layer{layer}.npz')
    for ub in UB_VALUES:
        suffix = '' if ub == 5 else f'_ub{ub}'
        for layer in [0, 1]:
            required.append(f'intensity_layer{layer}{suffix}.npz')
    return all(os.path.exists(os.path.join(tmp, f)) for f in required)


# ─── Plot Assembly Functions ───────────────────────────────────────────

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
        ax.plot(pos[valid], cond_acc[valid], marker='s', markersize=3, linewidth=1.2, color='#e6850e')
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
    fig.savefig(os.path.join(plot_dir, 'baseline_conditional_accuracy.png'), dpi=300, bbox_inches='tight')
    plt.close()


def _assemble_ablation(tmp_dir, plot_dir, tag):
    data = {}
    for layer in [0, 1]:
        f = os.path.join(tmp_dir, f'ablation_layer{layer}.npz')
        if not os.path.exists(f):
            return
        d = np.load(f)
        data[layer] = {'full_seq_acc': float(d['full_seq_acc']),
                       'per_pos_acc': d['per_pos_acc'],
                       'cond_acc': d['cond_acc'], 'cond_eligible': d['cond_eligible']}

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
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
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
        ax.grid(True, alpha=0.3); ax.set_ylim(0, 1.05)
    axes[0].set_ylabel('Per-position accuracy', fontsize=10)
    fig.suptitle(f'Per-position accuracy with attention removed (500 trials)\n{tag}',
                 fontsize=11, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, 'ablation_per_position.png'), dpi=300, bbox_inches='tight')
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for i, layer in enumerate([0, 1]):
        ax = axes[i]
        ca = data[layer]['cond_acc']; ce = data[layer]['cond_eligible']
        pos = np.arange(len(ca)); valid = ce >= 10
        color = ['#1f77b4', '#ff7f0e'][i]
        if valid.any():
            ax.plot(pos[valid], ca[valid], marker='o', markersize=3, linewidth=1.2, color=color)
            if not valid.all():
                cutoff = np.where(~valid)[0][0]
                ax.axvline(x=cutoff - 0.5, color=color, linestyle=':', alpha=0.5)
        ax.set_xlabel('Output position', fontsize=10)
        ax.set_title(f'Skip Layer {layer}', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3); ax.set_ylim(0, 1.05)
    axes[0].set_ylabel('Conditional per-token accuracy', fontsize=10)
    fig.suptitle(f'Per-token accuracy (given prefix correct) with attention removed (500 trials)\n{tag}',
                 fontsize=11, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, 'ablation_conditional_accuracy.png'), dpi=300, bbox_inches='tight')
    plt.close()


def _assemble_cinclogits(tmp_dir, plot_dir, tag):
    for layer in [0, 1]:
        f = os.path.join(tmp_dir, f'cinclogits_layer{layer}.npz')
        if not os.path.exists(f):
            continue
        d = np.load(f)
        cl_ic = d['clogit_icscore']; icl_ic = d['iclogit_icscore']
        frac_ic = np.mean(cl_ic + icl_ic)
        eps = 1e-10
        corr = np.sum(cl_ic) / (np.sum(cl_ic + icl_ic) + eps)

        fig, ax = plt.subplots(figsize=(4.5, 4))
        bw = 0.5; x = np.array([0, 1])
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
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ymax = max(frac_ic, corr)
        ax.set_ylim(0, ymax * 1.25 if ymax > 0 else 1.0)
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, f'cinclogits_layer{layer}.png'), dpi=300, bbox_inches='tight')
        plt.close()


def _assemble_intensity(tmp_dir, plot_dir, tag):
    for ub in UB_VALUES:
        for layer in [0, 1]:
            suffix = '' if ub == 5 else f'_ub{ub}'
            f = os.path.join(tmp_dir, f'intensity_layer{layer}{suffix}.npz')
            if not os.path.exists(f):
                continue
            d = np.load(f)
            intensities = d['intensities']; rates = d['success_rates']
            plt.figure(figsize=(4.5, 3.2))
            plt.plot(intensities, rates, marker='o', linewidth=1.5, markersize=5, color='#e6850e')
            plt.xlabel('Intervention Intensity', fontsize=9)
            plt.ylabel('Success Probability', fontsize=9)
            title = f'Robustness to Attention Intervention (Layer {layer})'
            if ub != 5:
                title += f'  [ub={ub}]'
            title += f'\n{tag}'
            plt.title(title, fontsize=9)
            plt.grid(True, alpha=0.3)
            plt.xticks(intensities[::2], fontsize=8); plt.yticks(fontsize=8)
            plt.ylim(0, 1.05)
            plt.tight_layout()
            fname = f'intensity_layer{layer}' + (f'_ub{ub}' if ub != 5 else '')
            plt.savefig(os.path.join(plot_dir, f'{fname}.png'), dpi=300, bbox_inches='tight')
            plt.close()


def _assemble_asymmetric(tmp_dir, plot_dir, tag):
    f_ub = os.path.join(tmp_dir, 'intensity_layer0_ub60_lb0.npz')
    f_lb = os.path.join(tmp_dir, 'intensity_layer0_ub0_lb60.npz')
    if not (os.path.exists(f_ub) and os.path.exists(f_lb)):
        return
    d_ub = np.load(f_ub); d_lb = np.load(f_lb)

    plt.figure(figsize=(5.5, 3.8))
    plt.plot(d_ub['intensities'], d_ub['success_rates'],
             marker='o', linewidth=1.8, markersize=6,
             label='ub=60, lb=0 (above target)', color='#e6850e')
    plt.plot(d_lb['intensities'], d_lb['success_rates'],
             marker='s', linewidth=1.8, markersize=6,
             label='ub=0, lb=60 (below target)', color='#1f77b4')
    plt.xlabel('Intervention Intensity', fontsize=10)
    plt.ylabel('Success Probability', fontsize=10)
    plt.title(f'Asymmetric Intervention Robustness (Layer 0)\n{tag}',
              fontsize=11, fontweight='bold')
    plt.legend(fontsize=9, loc='lower left')
    plt.grid(True, alpha=0.3)
    plt.xticks(d_ub['intensities'], fontsize=9); plt.yticks(fontsize=9)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'intensity_layer0_asym_ub60_lb60.png'),
                dpi=300, bbox_inches='tight')
    plt.close()


def _assemble_hijack(tmp_dir, plot_dir, tag):
    f = os.path.join(tmp_dir, 'hijack.npz')
    if not os.path.exists(f):
        return
    d = np.load(f)
    data = d['data']
    if len(data) == 0:
        return

    current = data[:, 0]; boosted = data[:, 1]
    predicted = data[:, 2]; correct = data[:, 3]
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

    for arr, cmap, label, fname in [
        (break_map, 'YlOrRd', 'Breaking Rate', 'hijack_breaking_rate_heatmap_layer0.png'),
        (hijack_map, 'YlOrRd', 'Hijack Rate', 'hijack_hijack_rate_heatmap_layer0.png'),
    ]:
        fig, ax = plt.subplots(figsize=(10, 8.5))
        im = ax.imshow(arr, aspect='auto', cmap=cmap, vmin=0, vmax=1,
                       interpolation='nearest', origin='lower')
        ax.set_xlabel('Intervened-toward Number (binned)', fontsize=12)
        ax.set_ylabel('Current Number (binned)', fontsize=12)
        title_map = {'Breaking Rate': f'Breaking Rate: P(pred ≠ correct)',
                     'Hijack Rate': f'Hijack Rate: P(pred == intervened target)'}
        ax.set_title(f'{title_map[label]}\n{tag}  intensity=10', fontsize=12, fontweight='bold')
        ax.set_xticks(tick_positions); ax.set_xticklabels(tick_labels, fontsize=8)
        ax.set_yticks(tick_positions); ax.set_yticklabels(tick_labels, fontsize=8)
        plt.colorbar(im, ax=ax, label=label, shrink=0.85)
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, fname), dpi=200, bbox_inches='tight')
        plt.close()

    fig, ax = plt.subplots(figsize=(10, 8.5))
    im = ax.imshow(count_map, aspect='auto', cmap='viridis', interpolation='nearest', origin='lower')
    ax.set_xlabel('Intervened-toward Number (binned)', fontsize=12)
    ax.set_ylabel('Current Number (binned)', fontsize=12)
    ax.set_title(f'Sample Count per (current, target) bin\n{tag}  intensity=10',
                 fontsize=11, fontweight='bold')
    ax.set_xticks(tick_positions); ax.set_xticklabels(tick_labels, fontsize=8)
    ax.set_yticks(tick_positions); ax.set_yticklabels(tick_labels, fontsize=8)
    plt.colorbar(im, ax=ax, label='Count', shrink=0.85)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, 'hijack_sample_count_heatmap_layer0.png'),
                dpi=200, bbox_inches='tight')
    plt.close()


def _assemble_separator_random(tmp_dir, plot_dir, tag):
    f = os.path.join(tmp_dir, 'separator_random.npz')
    if not os.path.exists(f):
        return
    d = np.load(f)
    sep_data = d['sep_data']; rand_data = d['rand_data']

    for data, title_prefix, filename in [
        (sep_data, 'Intervention Success when Attending to Separator',
         'intervention_pernumber_separator_layer0.png'),
        (rand_data, 'Intervention Success with Random Target',
         'intervention_pernumber_random_layer0.png'),
    ]:
        if len(data) == 0:
            continue
        colors = {2.0: '#1f77b4', 6.0: '#ff7f0e', 10.0: '#d62728'}
        fig, axes = plt.subplots(2, 1, figsize=(14, 8),
                                 gridspec_kw={'height_ratios': [3, 1]})
        ax = axes[0]
        for intens in INTENSITIES_SEP:
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
        ax.grid(True, alpha=0.3); ax.set_ylim(-0.05, 1.1); ax.set_xlim(0, 255)

        ax2 = axes[1]
        max_intens = max(INTENSITIES_SEP)
        mask_hi = data[:, 1] == max_intens
        counts = np.array([(mask_hi & (data[:, 0] == n)).sum() for n in range(256)])
        ax2.bar(range(256), counts, width=1, color='#666', alpha=0.5)
        ax2.set_xlabel('Number in Vocabulary', fontsize=12)
        ax2.set_ylabel('Sample Count', fontsize=10)
        ax2.set_xlim(0, 255); ax2.grid(True, alpha=0.2)
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, filename), dpi=200, bbox_inches='tight')
        plt.close()


def assemble_plots(ckpt):
    folder_name = ckpt['folder_name']
    tmp_dir = os.path.join(OUTPUT_BASE, 'tmp_results', folder_name)
    plot_dir = os.path.join(OUTPUT_BASE, folder_name)
    os.makedirs(plot_dir, exist_ok=True)
    tag = (f"V={ckpt['vocab']}  B={ckpt['block']}  lr={ckpt['lr']}  "
           f"iters={ckpt['itr']}  dseed={ckpt['dseed']}  iseed={ckpt['iseed']}")

    for fn, label in [
        (_assemble_baseline, 'baseline'), (_assemble_ablation, 'ablation'),
        (_assemble_cinclogits, 'cinclogits'), (_assemble_intensity, 'intensity'),
        (_assemble_asymmetric, 'asymmetric'), (_assemble_hijack, 'hijack'),
        (_assemble_separator_random, 'sep_rand'),
    ]:
        try:
            fn(tmp_dir, plot_dir, tag)
        except Exception as e:
            print(f"  WARN {label} for {folder_name}: {e}", flush=True)

    n_plots = len([f for f in os.listdir(plot_dir) if f.endswith('.png')])
    return n_plots


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

    assembled = set()
    for ckpt in checkpoints:
        if is_ckpt_done(ckpt):
            n = assemble_plots(ckpt)
            assembled.add(ckpt['folder_name'])
            print(f"  [PLOTS] {ckpt['folder_name']}: {n} plots (cached)")

    if not to_run:
        print("\nAll done!")
        return

    gpu_tasks = {g: [] for g in range(NUM_GPUS)}
    path_to_gpu = {}
    for i, ckpt in enumerate(checkpoints):
        path_to_gpu[ckpt['path']] = i % NUM_GPUS

    for t in to_run:
        g = path_to_gpu.get(t['ckpt_path'], hash(t['ckpt_path']) % NUM_GPUS)
        gpu_tasks[g].append(t)

    for g in gpu_tasks:
        gpu_tasks[g].sort(key=lambda t: (t['ckpt_path'], t['type'], t.get('layer', 0)))

    print(f"\nDistributed {len(to_run)} tasks across {NUM_GPUS} GPUs:")
    for g in range(NUM_GPUS):
        n = len(gpu_tasks[g])
        ckpts = len(set(t['ckpt_path'] for t in gpu_tasks[g])) if n else 0
        print(f"  GPU {g}: {n} tasks across {ckpts} checkpoints")

    task_dir = os.path.join(OUTPUT_BASE, 'task_files')
    log_dir = os.path.join(OUTPUT_BASE, 'worker_logs')
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
            [sys.executable, os.path.join(SCRIPT_DIR, 'gpu_worker.py'),
             '--tasks-file', tf, '--gpu', str(g)],
            stdout=lf, stderr=subprocess.STDOUT, cwd=SCRIPT_DIR)
        procs[g] = (proc, lf)

    print(f"\nLaunched {len(procs)} workers. Monitoring...\n", flush=True)

    last_print = 0
    while any(p.poll() is None for p, _ in procs.values()):
        time.sleep(10)
        for ckpt in checkpoints:
            fn = ckpt['folder_name']
            if fn not in assembled and is_ckpt_done(ckpt):
                n = assemble_plots(ckpt)
                assembled.add(fn)
                elapsed = time.time() - t_start
                print(f"  [PLOTS] {fn}: {n} plots ({elapsed:.0f}s)", flush=True)

        done_now = sum(1 for t in all_tasks if os.path.exists(t['out']))
        elapsed = time.time() - t_start
        if done_now >= last_print + 20:
            last_print = done_now
            rate = done_now / elapsed if elapsed > 0 else 0
            eta = (len(all_tasks) - done_now) / rate if rate > 0 else 0
            print(f"  [PROGRESS] {done_now}/{len(all_tasks)} tasks, "
                  f"{len(assembled)}/{len(checkpoints)} ckpts plotted "
                  f"({elapsed:.0f}s, ETA ~{eta:.0f}s)", flush=True)

    for g, (proc, lf) in procs.items():
        lf.close()
        if proc.returncode != 0:
            print(f"  [WARN] GPU {g} exited with code {proc.returncode}", flush=True)

    for ckpt in checkpoints:
        fn = ckpt['folder_name']
        if fn not in assembled:
            n = assemble_plots(ckpt)
            assembled.add(fn)
            print(f"  [PLOTS] {fn}: {n} plots (final)", flush=True)

    total_elapsed = time.time() - t_start
    total_plots = sum(len([f for f in os.listdir(os.path.join(OUTPUT_BASE, ckpt['folder_name']))
                           if f.endswith('.png')])
                      for ckpt in checkpoints
                      if os.path.isdir(os.path.join(OUTPUT_BASE, ckpt['folder_name'])))

    print(f"\n{'='*60}")
    print(f"ALL DONE — {len(assembled)}/{len(checkpoints)} checkpoints, {total_plots} plots total")
    print(f"Elapsed: {total_elapsed:.0f}s ({total_elapsed/60:.1f}m)")
    print(f"Output: {OUTPUT_BASE}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
