#!/usr/bin/env python3
"""
Generate publication figure for the residual stream cleanup results.

Two-panel figure:
  Left:  Per-checkpoint bar chart of Claim 1 & Claim 2 agreement rates
  Right: Schematic / summary scatter showing the two claims
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

RESULTS = os.path.join(os.path.dirname(__file__), 'residual_removal_results.json')
OUTDIR = os.path.join(os.path.dirname(__file__), 'plots')
os.makedirs(OUTDIR, exist_ok=True)

with open(RESULTS) as f:
    data = json.load(f)

leaps = [r for r in data if r['is_leapformer']]
leaps.sort(key=lambda r: (r['config'], r['seed']))

labels = [r['label'].replace('_', r'\_') for r in leaps]
c1 = [r['claim1_agree_pct'] for r in leaps]
c2 = [r['claim2_agree_pct'] for r in leaps]

# ── Identify the two known outliers (k16_N1024_s3, k32_N1024_s5) ──
outlier_labels = {'k16_N1024_s3', 'k32_N1024_s5'}
is_outlier = [r['label'] in outlier_labels for r in leaps]

# Also tag models where claim2 fails (< 80%) but aren't the known outliers
c2_fail = [(v < 80 and not o) for v, o in zip(c2, is_outlier)]

plt.rcParams.update({
    'font.size': 9, 'axes.labelsize': 11, 'axes.titlesize': 12,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2),
                                gridspec_kw={'width_ratios': [2.2, 1]})

# ── Left panel: grouped bar chart ──
x = np.arange(len(leaps))
w = 0.35

bars1 = ax1.bar(x - w/2, c1, w, color='#2a9d8f', edgecolor='white',
                linewidth=0.5, label=r'Claim 1: remove $\mathbf{a}_1$ after MLP1', zorder=3)
bars2 = ax1.bar(x + w/2, c2, w, color='#e76f51', edgecolor='white',
                linewidth=0.5, label=r'Claim 2: remove $\mathbf{m}_1$ at readout', zorder=3)

for i, (o, f) in enumerate(zip(is_outlier, c2_fail)):
    if o:
        ax1.bar(x[i] + w/2, c2[i], w, color='#e76f51', edgecolor='#264653',
                linewidth=1.5, hatch='///', zorder=4)
    elif f:
        ax1.bar(x[i] + w/2, c2[i], w, color='#e76f51', edgecolor='#264653',
                linewidth=1.5, hatch='...', zorder=4)

ax1.axhline(100, color='gray', linewidth=0.6, linestyle='--', alpha=0.5, zorder=1)
ax1.axhline(90, color='gray', linewidth=0.4, linestyle=':', alpha=0.4, zorder=1)
ax1.set_xticks(x)
short_labels = []
for r in leaps:
    cfg = r['config'].replace('k', '').replace('_N', ',')
    short_labels.append(f"{cfg} s{r['seed']}")
ax1.set_xticklabels(short_labels, rotation=55, ha='right', fontsize=7.5)
ax1.set_ylabel('Argmax agreement with normal forward (%)')
ax1.set_ylim(0, 108)
ax1.set_title('Residual stream component removal: per-checkpoint results')
ax1.legend(loc='lower left', fontsize=8, framealpha=0.9)
ax1.grid(axis='y', alpha=0.2)

hatched_patch = mpatches.Patch(facecolor='#e76f51', edgecolor='#264653',
                                hatch='///', label='Known outlier')
dotted_patch = mpatches.Patch(facecolor='#e76f51', edgecolor='#264653',
                               hatch='...', label='Claim 2 failure')
ax1.legend(handles=[bars1, bars2, hatched_patch, dotted_patch],
           loc='lower left', fontsize=7.5, framealpha=0.9)

# ── Right panel: scatter of Claim1 vs Claim2 ──
non_outlier = [not o for o in is_outlier]
c1_no = [v for v, m in zip(c1, non_outlier) if m]
c2_no = [v for v, m in zip(c2, non_outlier) if m]
c1_ol = [v for v, m in zip(c1, is_outlier) if m]
c2_ol = [v for v, m in zip(c2, is_outlier) if m]

ax2.scatter(c1_no, c2_no, s=60, c='#2a9d8f', edgecolors='#264653',
            linewidth=0.8, zorder=3, label='Standard leap-formers')
ax2.scatter(c1_ol, c2_ol, s=80, c='#e76f51', edgecolors='#264653',
            linewidth=0.8, marker='X', zorder=4, label='Known outliers')

for r in leaps:
    if r['label'] in outlier_labels:
        cfg = r['config'].replace('k', '').replace('_N', ',')
        lbl = f"{cfg} s{r['seed']}"
        ax2.annotate(lbl, (r['claim1_agree_pct'], r['claim2_agree_pct']),
                     textcoords='offset points', xytext=(6, -8), fontsize=7,
                     color='#264653')

ax2.axhline(90, color='#e76f51', linewidth=0.8, linestyle=':', alpha=0.6)
ax2.axvline(90, color='#2a9d8f', linewidth=0.8, linestyle=':', alpha=0.6)
ax2.fill_between([90, 101], 90, 101, alpha=0.08, color='#2a9d8f')

ax2.set_xlabel(r'Claim 1 agreement (%)')
ax2.set_ylabel(r'Claim 2 agreement (%)')
ax2.set_xlim(0, 102)
ax2.set_ylim(0, 102)
ax2.set_title('Claim 1 vs Claim 2')
ax2.legend(fontsize=8, loc='lower left')
ax2.grid(alpha=0.2)
ax2.set_aspect('equal')

fig.tight_layout()
out = os.path.join(OUTDIR, 'residual_stream_cleanup.png')
fig.savefig(out, dpi=250, bbox_inches='tight')
plt.close()
print(f"Saved {out}")
