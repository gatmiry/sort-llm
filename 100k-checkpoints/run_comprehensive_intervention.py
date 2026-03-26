#!/usr/bin/env python3
"""
Comprehensive Layer-0 Intervention Analysis for 100k checkpoints.
  1. Discovers all 24 checkpoints in final_models/
  2. Distributes across 8 GPUs (3 checkpoints per GPU)
  3. Each GPU worker generates 3000 random sequences per checkpoint,
     intervenes at every sorted position with 4 intensities [2,4,6,10]
  4. After workers finish, loads all data and generates:
     - Number vulnerability curves
     - Position vulnerability curves
     - Gap analysis
     - Number × Position heatmap
     - Training progression comparison
     - Error direction analysis
     - Seed consistency plots
     - Detailed text summary with pattern analysis
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
CKPT_DIR = os.path.join(SCRIPT_DIR, 'final_models')
OUTPUT_BASE = os.path.join(SCRIPT_DIR, 'outputs', 'comprehensive_intervention')
RAW_DIR = os.path.join(OUTPUT_BASE, 'raw')
PLOT_DIR = os.path.join(OUTPUT_BASE, 'plots')
LOG_DIR = os.path.join(OUTPUT_BASE, 'logs')
TASK_DIR = os.path.join(OUTPUT_BASE, 'tasks')
NUM_GPUS = 8


# ─── checkpoint discovery ───────────────────────────────────────────────
def discover_checkpoints():
    pts = sorted(glob.glob(os.path.join(CKPT_DIR, '*.pt')))
    ckpts = []
    for pt in pts:
        bn = os.path.basename(pt)
        if '.summary.' in bn:
            continue
        parts = bn.replace('.pt', '').split('__')
        config_str, stage = parts[0], (parts[1] if len(parts) > 1 else 'final')
        tokens = config_str.split('_')
        dseed = iseed = None
        for tok in tokens:
            if tok.startswith('dseed'):
                dseed = tok[5:]
            elif tok.startswith('iseed'):
                iseed = tok[5:]
        if stage.startswith('ckpt'):
            itr = int(stage[4:])
            label = stage
        else:
            itr = 100000
            label = 'final'
        name = f"ds{dseed}_is{iseed}_{label}"
        ckpts.append(dict(path=pt, dseed=dseed, iseed=iseed,
                          itr=itr, label=label, name=name))
    return ckpts


# ─── launch / monitor workers ───────────────────────────────────────────
def launch_workers(ckpts):
    for d in [RAW_DIR, PLOT_DIR, LOG_DIR, TASK_DIR]:
        os.makedirs(d, exist_ok=True)

    gpu_tasks = {g: [] for g in range(NUM_GPUS)}
    for i, c in enumerate(ckpts):
        g = i % NUM_GPUS
        gpu_tasks[g].append(dict(
            ckpt_path=c['path'], name=c['name'],
            out=os.path.join(RAW_DIR, f"{c['name']}.npz")))

    procs = {}
    for g in range(NUM_GPUS):
        if not gpu_tasks[g]:
            continue
        tf = os.path.join(TASK_DIR, f'gpu{g}.json')
        with open(tf, 'w') as f:
            json.dump(gpu_tasks[g], f)
        lf = open(os.path.join(LOG_DIR, f'gpu{g}.log'), 'w')
        proc = subprocess.Popen(
            [sys.executable, os.path.join(SCRIPT_DIR, 'comprehensive_worker.py'),
             '--tasks-file', tf, '--gpu', str(g)],
            stdout=lf, stderr=subprocess.STDOUT, cwd=SCRIPT_DIR)
        procs[g] = proc
    return procs


def wait_for_workers(procs, ckpts):
    t0 = time.time()
    while any(p.poll() is None for p in procs.values()):
        time.sleep(10)
        done = sum(1 for c in ckpts
                   if os.path.exists(os.path.join(RAW_DIR, f"{c['name']}.npz")))
        elapsed = time.time() - t0
        print(f"  [{elapsed:.0f}s] {done}/{len(ckpts)} checkpoints done", flush=True)
    elapsed = time.time() - t0
    done = sum(1 for c in ckpts
               if os.path.exists(os.path.join(RAW_DIR, f"{c['name']}.npz")))
    print(f"Workers finished: {done}/{len(ckpts)} in {elapsed:.0f}s", flush=True)
    for g, p in procs.items():
        if p.returncode != 0:
            print(f"  WARN: GPU {g} exit code {p.returncode}", flush=True)


# ─── load data ───────────────────────────────────────────────────────────
def load_all_data(ckpts):
    arrays, meta = [], []
    for c in ckpts:
        f = os.path.join(RAW_DIR, f"{c['name']}.npz")
        if not os.path.exists(f):
            continue
        d = np.load(f)
        n = len(d['position'])
        arrays.append({k: d[k] for k in d.files})
        meta.append(dict(name=c['name'], dseed=c['dseed'], iseed=c['iseed'],
                         itr=c['itr'], label=c['label'], n=n))

    if not arrays:
        print("ERROR: no data files found!")
        return None, None

    combined = {}
    for key in ['position', 'number', 'next_number', 'gap',
                'intensity', 'correct', 'predicted']:
        combined[key] = np.concatenate([a[key] for a in arrays])
    combined['ckpt_idx'] = np.concatenate(
        [np.full(m['n'], i, dtype=np.int16) for i, m in enumerate(meta)])
    return combined, meta


# ─── analysis & plotting ─────────────────────────────────────────────────
def analyze_and_plot(combined, meta):
    pos = combined['position']
    num = combined['number']
    nxt = combined['next_number']
    gap = combined['gap']
    intens = combined['intensity']
    correct = combined['correct'].astype(np.float64)
    predicted = combined['predicted']
    cidx = combined['ckpt_idx']

    all_intens = sorted(set(intens.tolist()))
    nonzero_intens = [v for v in all_intens if v > 0]
    max_intens = max(nonzero_intens)

    C = {0.0: '#2ca02c', 2.0: '#1f77b4', 4.0: '#ff7f0e',
         6.0: '#d62728', 10.0: '#9467bd'}
    summary = []
    summary.append("=" * 72)
    summary.append("COMPREHENSIVE INTERVENTION ANALYSIS — LAYER 0")
    summary.append(f"Data points: {len(pos):,}   Checkpoints: {len(meta)}")
    summary.append(f"Intensities tested: {all_intens}")
    summary.append("=" * 72)

    # ── 1. Success rate by NUMBER ──────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(15, 9), gridspec_kw={'height_ratios': [3, 1]})

    ax = axes[0]
    for iv in all_intens:
        mask = intens == iv
        xs, ys = [], []
        for n_val in range(256):
            m = mask & (num == n_val)
            if m.sum() >= 20:
                xs.append(n_val)
                ys.append(correct[m].mean())
        lbl = 'baseline' if iv == 0 else f'intensity={iv}'
        ax.plot(xs, ys, color=C.get(iv, '#333'), label=lbl, linewidth=0.8, alpha=0.85)

    # smoothed overlay for highest intensity
    mask_hi = intens == max_intens
    raw_y = np.full(256, np.nan)
    for n_val in range(256):
        m = mask_hi & (num == n_val)
        if m.sum() >= 10:
            raw_y[n_val] = correct[m].mean()
    win = 11
    pad = win // 2
    smoothed = np.convolve(np.nan_to_num(raw_y, nan=0.5),
                           np.ones(win) / win, mode='same')
    valid = ~np.isnan(raw_y)
    ax.plot(np.arange(256)[valid], smoothed[valid],
            color=C[max_intens], linewidth=3, linestyle='--',
            label=f'smoothed int={max_intens}', alpha=0.9)
    ax.set_ylabel('Success Rate', fontsize=12)
    ax.set_title('Intervention Success Rate by Number (Layer 0, all checkpoints)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=8, ncol=3, loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.1)
    ax.set_xlim(0, 255)

    # sample count per number (at max intensity)
    ax2 = axes[1]
    counts_per_num = np.array([((intens == max_intens) & (num == n)).sum()
                                for n in range(256)])
    ax2.bar(range(256), counts_per_num, width=1, color='#666', alpha=0.5)
    ax2.set_xlabel('Number', fontsize=12)
    ax2.set_ylabel('Sample count', fontsize=10)
    ax2.set_xlim(0, 255)
    ax2.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, 'success_by_number.png'), dpi=200)
    plt.close()

    # ── 2. Success rate by POSITION ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for iv in all_intens:
        mask = intens == iv
        xs, ys, errs = [], [], []
        for p in range(15):
            m = mask & (pos == p)
            if m.sum() >= 50:
                xs.append(p)
                mean = correct[m].mean()
                ys.append(mean)
                se = np.sqrt(mean * (1 - mean) / m.sum())
                errs.append(1.96 * se)
        lbl = 'baseline' if iv == 0 else f'intensity={iv}'
        ax.errorbar(xs, ys, yerr=errs, marker='o', color=C.get(iv, '#333'),
                    label=lbl, linewidth=2, markersize=5, capsize=3)
    ax.set_xlabel('Sorted Position (0 = smallest number)', fontsize=12)
    ax.set_ylabel('Success Rate', fontsize=12)
    ax.set_title('Intervention Success Rate by Sorted Position (Layer 0)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.1)
    ax.set_xticks(range(15))
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, 'success_by_position.png'), dpi=200)
    plt.close()

    # ── 3. Success rate by GAP ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for iv in nonzero_intens:
        mask = intens == iv
        xs, ys = [], []
        for g_lo in range(1, 60, 2):
            m = mask & (gap >= g_lo) & (gap < g_lo + 2)
            if m.sum() >= 30:
                xs.append(g_lo + 1)
                ys.append(correct[m].mean())
        ax.plot(xs, ys, marker='.', color=C.get(iv, '#333'),
                label=f'intensity={iv}', linewidth=1.5)
    ax.set_xlabel('Gap to Next Sorted Number', fontsize=12)
    ax.set_ylabel('Success Rate', fontsize=12)
    ax.set_title('Intervention Success Rate by Gap Size (Layer 0)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.1)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, 'success_by_gap.png'), dpi=200)
    plt.close()

    # ── 4. Number × Position heatmap (highest intensity) ──────────────
    mask_hi = intens == max_intens
    n_bins = 32
    bin_size = 256 // n_bins
    hmap = np.full((15, n_bins), np.nan)
    hmap_cnt = np.zeros((15, n_bins), dtype=int)
    for p in range(15):
        for b in range(n_bins):
            lo, hi = b * bin_size, (b + 1) * bin_size
            m = mask_hi & (pos == p) & (num >= lo) & (num < hi)
            cnt = int(m.sum())
            hmap_cnt[p, b] = cnt
            if cnt >= 5:
                hmap[p, b] = correct[m].mean()

    fig, ax = plt.subplots(figsize=(14, 6))
    im_obj = ax.imshow(hmap, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1,
                       interpolation='nearest', origin='lower',
                       extent=[0, 256, -0.5, 14.5])
    ax.set_xlabel('Number (binned)', fontsize=12)
    ax.set_ylabel('Sorted Position', fontsize=12)
    ax.set_title(f'Success Rate Heatmap (Layer 0, intensity={max_intens})',
                 fontsize=13, fontweight='bold')
    ax.set_yticks(range(15))
    plt.colorbar(im_obj, ax=ax, label='Success Rate')
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, 'heatmap_number_position.png'), dpi=200)
    plt.close()

    # full-resolution heatmap
    hmap_full = np.full((15, 256), np.nan)
    for p in range(15):
        for n_val in range(256):
            m = mask_hi & (pos == p) & (num == n_val)
            if m.sum() >= 3:
                hmap_full[p, n_val] = correct[m].mean()
    fig, ax = plt.subplots(figsize=(18, 6))
    im_obj = ax.imshow(hmap_full, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1,
                       interpolation='nearest', origin='lower')
    ax.set_xlabel('Number', fontsize=12)
    ax.set_ylabel('Sorted Position', fontsize=12)
    ax.set_title(f'Success Rate — Full Resolution (Layer 0, intensity={max_intens})',
                 fontsize=13, fontweight='bold')
    ax.set_yticks(range(15))
    plt.colorbar(im_obj, ax=ax, label='Success Rate')
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, 'heatmap_full_resolution.png'), dpi=200)
    plt.close()

    # ── 5. Training progression ───────────────────────────────────────
    stages = [('ckpt60000', 60000), ('ckpt80000', 80000), ('final', 100000)]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for si, (stage, sitr) in enumerate(stages):
        ax = axes[si]
        sc = [i for i, m in enumerate(meta) if m['label'] == stage]
        if not sc:
            continue
        smask = np.isin(cidx, sc)
        for iv in [0.0] + nonzero_intens:
            mask2 = smask & (intens == iv)
            ns, cors = num[mask2], correct[mask2]
            xs, ys = [], []
            for n_lo in range(0, 256, 4):
                m = (ns >= n_lo) & (ns < n_lo + 4)
                if m.sum() >= 5:
                    xs.append(n_lo + 2)
                    ys.append(cors[m].mean())
            lbl = 'baseline' if iv == 0 else f'int={iv}'
            ax.plot(xs, ys, color=C.get(iv, '#333'), label=lbl,
                    linewidth=1, alpha=0.85)
        ax.set_title(f'{stage} ({sitr} iters)', fontsize=11, fontweight='bold')
        ax.set_xlabel('Number', fontsize=10)
        if si == 0:
            ax.set_ylabel('Success Rate', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.1)
        ax.legend(fontsize=7)
    fig.suptitle('Training Progression: Number Vulnerability (Layer 0)',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, 'training_progression_number.png'), dpi=200)
    plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for si, (stage, sitr) in enumerate(stages):
        ax = axes[si]
        sc = [i for i, m in enumerate(meta) if m['label'] == stage]
        if not sc:
            continue
        smask = np.isin(cidx, sc)
        for iv in [0.0] + nonzero_intens:
            mask2 = smask & (intens == iv)
            ps, cors = pos[mask2], correct[mask2]
            xs, ys = [], []
            for pv in range(15):
                m = ps == pv
                if m.sum() >= 20:
                    xs.append(pv)
                    ys.append(cors[m].mean())
            lbl = 'baseline' if iv == 0 else f'int={iv}'
            ax.plot(xs, ys, marker='o', color=C.get(iv, '#333'), label=lbl,
                    linewidth=1.5, markersize=4)
        ax.set_title(f'{stage}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Sorted Position', fontsize=10)
        if si == 0:
            ax.set_ylabel('Success Rate', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.1)
        ax.legend(fontsize=7, loc='lower left')
    fig.suptitle('Training Progression: Position Vulnerability (Layer 0)',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, 'training_progression_position.png'), dpi=200)
    plt.close()

    # ── 6. Error analysis ─────────────────────────────────────────────
    wrong_mask = (intens == max_intens) & (correct == 0)
    n_wrong = int(wrong_mask.sum())
    if n_wrong > 0:
        wp = predicted[wrong_mask].astype(np.int32)
        wn = nxt[wrong_mask].astype(np.int32)
        wnum = num[wrong_mask].astype(np.int32)
        delta = wp - wn

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        ax = axes[0]
        bins = np.arange(-60, 62, 2)
        ax.hist(np.clip(delta, -60, 60), bins=bins,
                color='#d62728', alpha=0.7, edgecolor='black', linewidth=0.3)
        ax.axvline(0, color='black', linewidth=1, linestyle='--')
        ax.set_xlabel('Prediction Error (pred − correct)', fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        ax.set_title(f'Error Direction (int={max_intens})', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        n_show = min(8000, n_wrong)
        idx_show = np.random.choice(n_wrong, n_show, replace=False)
        ax.scatter(wn[idx_show], wp[idx_show], s=1, alpha=0.25, c='#d62728')
        ax.plot([0, 256], [0, 256], 'k--', linewidth=1)
        ax.set_xlabel('Correct Number', fontsize=10)
        ax.set_ylabel('Predicted Number', fontsize=10)
        ax.set_title('Predicted vs Correct', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 256)
        ax.set_ylim(0, 256)

        ax = axes[2]
        pred_minus_cur = wp - wnum
        bins2 = np.arange(-20, 80, 2)
        ax.hist(np.clip(pred_minus_cur, -20, 80), bins=bins2,
                color='#ff7f0e', alpha=0.7, edgecolor='black', linewidth=0.3)
        ax.axvline(0, color='black', linewidth=1, linestyle='--')
        ax.set_xlabel('pred − current_number', fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        ax.set_title('Predicted relative to current', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(os.path.join(PLOT_DIR, 'error_analysis.png'), dpi=200)
        plt.close()

        # When wrong: is the predicted number the "intervened" number?
        # The intervention boosts a number > current within ub=60
        above_current = (wp > wnum) & (wp <= wnum + 60) & (wp != wn)
        pct_intervened = above_current.mean() * 100
        summary.append(f"\nError analysis (intensity={max_intens}):")
        summary.append(f"  Total errors: {n_wrong}")
        summary.append(f"  Mean error delta (pred−correct): {delta.mean():.2f}")
        summary.append(f"  Median error delta: {np.median(delta):.1f}")
        summary.append(f"  Pred too high: {(delta > 0).sum()} ({(delta > 0).mean()*100:.1f}%)")
        summary.append(f"  Pred too low:  {(delta < 0).sum()} ({(delta < 0).mean()*100:.1f}%)")
        summary.append(f"  Pred in intervened range (>cur, ≤cur+60, ≠correct): "
                        f"{int(above_current.sum())} ({pct_intervened:.1f}%)")

    # ── 7. Seed consistency ───────────────────────────────────────────
    dseeds = sorted(set(m['dseed'] for m in meta))
    iseeds = sorted(set(m['iseed'] for m in meta))

    fig, axes = plt.subplots(1, len(dseeds), figsize=(7 * len(dseeds), 5),
                             sharey=True, squeeze=False)
    for di, ds in enumerate(dseeds):
        ax = axes[0][di]
        ds_final = [i for i, m in enumerate(meta)
                    if m['dseed'] == ds and m['label'] == 'final']
        if not ds_final:
            continue
        ds_mask = np.isin(cidx, ds_final)
        for iv in nonzero_intens:
            m2 = ds_mask & (intens == iv)
            ns, cors = num[m2], correct[m2]
            xs, ys = [], []
            for n_lo in range(0, 256, 4):
                m = (ns >= n_lo) & (ns < n_lo + 4)
                if m.sum() >= 5:
                    xs.append(n_lo + 2)
                    ys.append(cors[m].mean())
            ax.plot(xs, ys, color=C.get(iv, '#333'),
                    label=f'int={iv}', linewidth=1, alpha=0.85)
        ax.set_title(f'dseed={ds} (final, all iseeds)', fontsize=11, fontweight='bold')
        ax.set_xlabel('Number', fontsize=10)
        if di == 0:
            ax.set_ylabel('Success Rate', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.1)
        ax.legend(fontsize=8)
    fig.suptitle('Data Seed Comparison (Layer 0, final checkpoints)',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, 'seed_comparison_dseed.png'), dpi=200)
    plt.close()

    # per-iseed overlay at final
    fig, ax = plt.subplots(figsize=(14, 5))
    for isi, isv in enumerate(iseeds):
        is_final = [i for i, m in enumerate(meta)
                    if m['iseed'] == isv and m['label'] == 'final']
        if not is_final:
            continue
        is_mask = np.isin(cidx, is_final)
        m2 = is_mask & (intens == max_intens)
        ns, cors = num[m2], correct[m2]
        xs, ys = [], []
        for n_lo in range(0, 256, 6):
            m = (ns >= n_lo) & (ns < n_lo + 6)
            if m.sum() >= 5:
                xs.append(n_lo + 3)
                ys.append(cors[m].mean())
        ax.plot(xs, ys, linewidth=1, alpha=0.8,
                label=f'iseed={isv}')
    ax.set_xlabel('Number', fontsize=12)
    ax.set_ylabel('Success Rate', fontsize=12)
    ax.set_title(f'Init Seed Consistency (Layer 0, intensity={max_intens}, final)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.1)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, 'seed_comparison_iseed.png'), dpi=200)
    plt.close()

    # ── 8. Combined summary figure ────────────────────────────────────
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

    # Panel A: number curve (smoothed)
    ax = fig.add_subplot(gs[0, :2])
    for iv in nonzero_intens:
        mask_iv = intens == iv
        raw_arr = np.full(256, np.nan)
        for n_val in range(256):
            m = mask_iv & (num == n_val)
            if m.sum() >= 10:
                raw_arr[n_val] = correct[m].mean()
        sm = np.convolve(np.nan_to_num(raw_arr, nan=0.5),
                         np.ones(11) / 11, mode='same')
        vld = ~np.isnan(raw_arr)
        ax.plot(np.arange(256)[vld], sm[vld], color=C.get(iv, '#333'),
                linewidth=2, label=f'int={iv}')
    ax.set_xlabel('Number')
    ax.set_ylabel('Success Rate')
    ax.set_title('A. Number Vulnerability (smoothed)', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.1)

    # Panel B: position curve
    ax = fig.add_subplot(gs[0, 2])
    for iv in nonzero_intens:
        mask_iv = intens == iv
        xs, ys = [], []
        for pv in range(15):
            m = mask_iv & (pos == pv)
            if m.sum() >= 50:
                xs.append(pv)
                ys.append(correct[m].mean())
        ax.plot(xs, ys, marker='o', color=C.get(iv, '#333'),
                linewidth=2, markersize=5, label=f'int={iv}')
    ax.set_xlabel('Sorted Position')
    ax.set_ylabel('Success Rate')
    ax.set_title('B. Position Vulnerability', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.1)

    # Panel C: heatmap
    ax = fig.add_subplot(gs[1, :2])
    im_obj = ax.imshow(hmap, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1,
                       interpolation='nearest', origin='lower',
                       extent=[0, 256, -0.5, 14.5])
    ax.set_xlabel('Number (binned)')
    ax.set_ylabel('Sorted Position')
    ax.set_title(f'C. Number × Position Heatmap (int={max_intens})', fontweight='bold')
    plt.colorbar(im_obj, ax=ax, label='Success Rate', shrink=0.8)

    # Panel D: gap analysis
    ax = fig.add_subplot(gs[1, 2])
    for iv in nonzero_intens:
        mask_iv = intens == iv
        xs, ys = [], []
        for g_lo in range(1, 50, 3):
            m = mask_iv & (gap >= g_lo) & (gap < g_lo + 3)
            if m.sum() >= 30:
                xs.append(g_lo + 1)
                ys.append(correct[m].mean())
        ax.plot(xs, ys, marker='.', color=C.get(iv, '#333'),
                label=f'int={iv}', linewidth=1.5)
    ax.set_xlabel('Gap')
    ax.set_ylabel('Success Rate')
    ax.set_title('D. Gap Analysis', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.1)

    # Panel E: error histogram
    if n_wrong > 0:
        ax = fig.add_subplot(gs[2, 0])
        bins = np.arange(-40, 42, 2)
        ax.hist(np.clip(delta, -40, 40), bins=bins,
                color='#d62728', alpha=0.7, edgecolor='black', linewidth=0.3)
        ax.axvline(0, color='black', linewidth=1, linestyle='--')
        ax.set_xlabel('Prediction Error')
        ax.set_title('E. Error Direction', fontweight='bold')
        ax.grid(True, alpha=0.3)

    # Panel F: training progression overlay (position, 3 stages)
    ax = fig.add_subplot(gs[2, 1:])
    markers = {'ckpt60000': 's', 'ckpt80000': '^', 'final': 'o'}
    for stage, sitr in stages:
        sc = [i for i, m in enumerate(meta) if m['label'] == stage]
        if not sc:
            continue
        smask = np.isin(cidx, sc)
        m2 = smask & (intens == max_intens)
        ps, cors = pos[m2], correct[m2]
        xs, ys = [], []
        for pv in range(15):
            m = ps == pv
            if m.sum() >= 20:
                xs.append(pv)
                ys.append(cors[m].mean())
        ax.plot(xs, ys, marker=markers[stage], linewidth=2, markersize=6,
                label=f'{stage} ({sitr})')
    ax.set_xlabel('Sorted Position')
    ax.set_ylabel('Success Rate')
    ax.set_title(f'F. Training Progression (int={max_intens})', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.1)

    fig.suptitle('Comprehensive Layer 0 Intervention Analysis — Summary',
                 fontsize=15, fontweight='bold', y=1.01)
    fig.savefig(os.path.join(PLOT_DIR, 'combined_summary.png'),
                dpi=200, bbox_inches='tight')
    plt.close()

    # ── 9. Numerical summary ──────────────────────────────────────────
    base_mask = intens == 0.0
    summary.append(f"\n--- Baseline (no intervention) ---")
    summary.append(f"Overall accuracy: {correct[base_mask].mean():.4f} "
                   f"(n={int(base_mask.sum())})")
    for pv in range(15):
        m = base_mask & (pos == pv)
        if m.sum() > 0:
            summary.append(f"  Position {pv:2d}: {correct[m].mean():.4f}")

    for iv in nonzero_intens:
        mask = intens == iv
        summary.append(f"\n--- Intensity = {iv} ---")
        summary.append(f"Total trials: {int(mask.sum()):,}")
        summary.append(f"Overall success: {correct[mask].mean():.4f}")

        num_rates = []
        for n_val in range(256):
            m = mask & (num == n_val)
            if m.sum() >= 15:
                num_rates.append((n_val, correct[m].mean(), int(m.sum())))
        num_rates.sort(key=lambda x: x[1])

        summary.append(f"\nTop 20 most vulnerable numbers:")
        for nv, rate, cnt in num_rates[:20]:
            summary.append(f"  #{nv:3d}: success={rate:.3f}  (n={cnt})")
        summary.append(f"\nTop 10 most robust numbers:")
        for nv, rate, cnt in num_rates[-10:]:
            summary.append(f"  #{nv:3d}: success={rate:.3f}  (n={cnt})")

        summary.append(f"\nBy position:")
        for pv in range(15):
            m = mask & (pos == pv)
            if m.sum() >= 50:
                summary.append(f"  Pos {pv:2d}: {correct[m].mean():.4f}  (n={int(m.sum())})")

        summary.append(f"\nBy gap range:")
        for gl, gh in [(1, 5), (5, 10), (10, 20), (20, 40), (40, 80), (80, 256)]:
            m = mask & (gap >= gl) & (gap < gh)
            if m.sum() >= 30:
                summary.append(f"  Gap [{gl:3d},{gh:3d}): {correct[m].mean():.4f}  (n={int(m.sum())})")

        lo = mask & (num < 20)
        hi = mask & (num > 235)
        mid = mask & (num >= 80) & (num <= 175)
        if lo.sum() >= 50:
            summary.append(f"\n  Edge low  (num<20):     {correct[lo].mean():.4f}  (n={int(lo.sum())})")
        if hi.sum() >= 50:
            summary.append(f"  Edge high (num>235):    {correct[hi].mean():.4f}  (n={int(hi.sum())})")
        if mid.sum() >= 50:
            summary.append(f"  Middle    (80≤num≤175): {correct[mid].mean():.4f}  (n={int(mid.sum())})")

    # training stages
    summary.append(f"\n--- Training Progression (int={max_intens}) ---")
    for stage, sitr in stages:
        sc = [i for i, m in enumerate(meta) if m['label'] == stage]
        if not sc:
            continue
        smask = np.isin(cidx, sc)
        m2 = smask & (intens == max_intens)
        if m2.sum() > 0:
            summary.append(f"  {stage:10s} ({sitr}): success={correct[m2].mean():.4f}  "
                           f"(n={int(m2.sum())})")

    # correlation: number value vs success
    mask_hiv = intens == max_intens
    valid_ns = []
    valid_rs = []
    for nv in range(256):
        m = mask_hiv & (num == nv)
        if m.sum() >= 10:
            valid_ns.append(nv)
            valid_rs.append(correct[m].mean())
    if len(valid_ns) >= 10:
        corr = np.corrcoef(valid_ns, valid_rs)[0, 1]
        summary.append(f"\nCorrelation (number vs success, int={max_intens}): r={corr:.4f}")

    # correlation: gap vs success
    mask_hiv2 = intens == max_intens
    valid_gs = []
    valid_grs = []
    for gv in range(1, 100):
        m = mask_hiv2 & (gap == gv)
        if m.sum() >= 20:
            valid_gs.append(gv)
            valid_grs.append(correct[m].mean())
    if len(valid_gs) >= 5:
        corr_g = np.corrcoef(valid_gs, valid_grs)[0, 1]
        summary.append(f"Correlation (gap vs success, int={max_intens}): r={corr_g:.4f}")

    summary_text = '\n'.join(summary)
    with open(os.path.join(OUTPUT_BASE, 'analysis_summary.txt'), 'w') as f:
        f.write(summary_text)
    print("\n" + summary_text)


# ─── main ────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    print("=" * 60)
    print("COMPREHENSIVE INTERVENTION ANALYSIS (Layer 0)")
    print("=" * 60)

    ckpts = discover_checkpoints()
    print(f"Found {len(ckpts)} checkpoints")

    done = sum(1 for c in ckpts
               if os.path.exists(os.path.join(RAW_DIR, f"{c['name']}.npz")))
    print(f"Already computed: {done}/{len(ckpts)}")

    if done < len(ckpts):
        procs = launch_workers(ckpts)
        print(f"Launched {len(procs)} GPU workers")
        wait_for_workers(procs, ckpts)

    print("\n" + "=" * 60)
    print("ANALYSIS PHASE")
    print("=" * 60)

    combined, meta = load_all_data(ckpts)
    if combined is not None:
        analyze_and_plot(combined, meta)

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s ({elapsed / 60:.1f}m)")


if __name__ == '__main__':
    main()
