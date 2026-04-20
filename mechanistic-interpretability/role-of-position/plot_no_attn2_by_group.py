#!/usr/bin/env python3
"""
Accuracy without attn2, broken down by i-group and gap.
Plots one curve per i-group, x-axis = gap, y-axis = accuracy.
"""
import os, sys, argparse, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'sortgpt_toolkit'))

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

from model import DEVICE, load_model_from_checkpoint

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str, required=True)
parser.add_argument('--out-tag', type=str, default='')
parser.add_argument('--n-batches', type=int, default=30000)
args = parser.parse_args()

OUTDIR = os.path.join(os.path.dirname(__file__), 'plots')
os.makedirs(OUTDIR, exist_ok=True)

GAPS = [1, 3, 5, 10, 20, 40]
GROUPS_BEGIN = {
    'i=1-3': list(range(1, 4)),
    'i=4-6': list(range(4, 7)),
    'i=7-9': list(range(7, 10)),
    'i=10-12': list(range(10, 13)),
    'i=13-15': list(range(13, 16)),
    'i=16-18': list(range(16, 19)),
}
GROUPS_END = {
    'i=477-497': list(range(477, 498)),
}
ALL_GROUPS = {**GROUPS_BEGIN, **GROUPS_END}
ALL_I = set()
for vs in ALL_GROUPS.values():
    ALL_I.update(vs)

print(f"Loading model from {args.ckpt} ...", flush=True)
model = load_model_from_checkpoint(args.ckpt)
bs = model.config.block_size
vn = model.config.vocab_size - 1
b0, b1 = model.transformer.h[0], model.transformer.h[1]


def get_batch():
    x = torch.randperm(vn)[:bs]
    vals, _ = torch.sort(x)
    sep = torch.tensor([vn])
    return torch.cat((x, sep, vals)).unsqueeze(0).to(DEVICE)


@torch.no_grad()
def collect_data():
    results = defaultdict(list)  # (cval, gap) -> list of bool
    t0 = time.time()

    for batch_num in range(args.n_batches):
        idx = get_batch()
        B, T = idx.size()
        sorted_vals = idx[0, bs + 1:]
        targets = sorted_vals[1:]

        pos = model.transformer.wpe(model.pos_idx[:T])
        embed = model.transformer.wte(idx) + pos
        x = b0(embed)
        x_no_a2 = x + b1.mlp(b1.ln_2(x))
        x_no_a2 = model.transformer.ln_f(x_no_a2)
        logits = x_no_a2 @ model.lm_head.weight.T
        preds = logits[0, bs + 1:2*bs].argmax(dim=-1)

        correct = (preds == targets).cpu().numpy()
        current_vals = sorted_vals[:-1].cpu().numpy()
        next_vals = targets.cpu().numpy()
        gaps = next_vals - current_vals

        for cv, g, c in zip(current_vals, gaps, correct):
            cv_int, g_int = int(cv), int(g)
            if cv_int in ALL_I and g_int in GAPS:
                results[(cv_int, g_int)].append(int(c))

        if (batch_num + 1) % 5000 == 0:
            total = sum(len(v) for v in results.values())
            el = time.time() - t0
            print(f"  batch {batch_num+1}: {total} samples ({el:.0f}s)", flush=True)

    return results


def compute_group_acc(results, group_ivals, gap):
    pooled = []
    for iv in group_ivals:
        pooled.extend(results.get((iv, gap), []))
    if len(pooled) < 10:
        return None, 0
    acc = np.mean(pooled)
    n_boot = 500
    boot = [np.mean(np.random.choice(pooled, size=len(pooled), replace=True))
            for _ in range(n_boot)]
    ci_lo = np.percentile(boot, 2.5)
    ci_hi = np.percentile(boot, 97.5)
    return (acc, ci_lo, ci_hi), len(pooled)


def plot_results(results):
    plt.rcParams.update({
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'savefig.facecolor': 'white', 'axes.spines.top': False,
        'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': 0.2,
        'font.size': 11,
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(GROUPS_BEGIN)))

    for ax, groups, title in [
        (ax1, GROUPS_BEGIN, 'Beginning of vocab'),
        (ax2, GROUPS_END, 'End of vocab'),
    ]:
        for idx_g, (gname, ivals) in enumerate(groups.items()):
            accs, ci_los, ci_his, valid_gaps, ns = [], [], [], [], []
            for gap in GAPS:
                res, n = compute_group_acc(results, ivals, gap)
                if res is not None:
                    accs.append(res[0] * 100)
                    ci_los.append(res[1] * 100)
                    ci_his.append(res[2] * 100)
                    valid_gaps.append(gap)
                    ns.append(n)

            if valid_gaps:
                c = colors[idx_g] if len(groups) > 1 else '#2166ac'
                ax.plot(valid_gaps, accs, 'o-', color=c, linewidth=2,
                        markersize=6, label=f'{gname} (n≈{np.mean(ns):.0f})')
                ax.fill_between(valid_gaps, ci_los, ci_his, color=c, alpha=0.15)

        ax.set_xlabel('Gap', fontsize=12)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xscale('log')
        ax.set_xticks(GAPS)
        ax.set_xticklabels([str(g) for g in GAPS])
        ax.legend(fontsize=9, loc='best')

    ax1.set_ylabel('Accuracy without attn2 (%)', fontsize=12)
    ax1.set_ylim(-5, 105)

    tag = f'_{args.out_tag}' if args.out_tag else ''
    fig.suptitle(f'Per-token accuracy after attn2 ablation{" (" + args.out_tag + ")" if args.out_tag else ""}',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    out_path = os.path.join(OUTDIR, f'no_attn2_acc_by_group{tag}.png')
    fig.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved {out_path}")


if __name__ == '__main__':
    print("Collecting data ...", flush=True)
    results = collect_data()

    print("\nResults summary:")
    for gname, ivals in ALL_GROUPS.items():
        for gap in GAPS:
            res, n = compute_group_acc(results, ivals, gap)
            if res:
                print(f"  {gname}, gap={gap}: acc={res[0]:.3f} (n={n})")
            else:
                print(f"  {gname}, gap={gap}: insufficient data (n={n})")

    plot_results(results)
    print("Done.")
