#!/usr/bin/env python3
"""
Combine per-seed hijack JSON data and plot average curves with confidence
intervals (mean +/- 1 std across seeds) for gap=1,10,20,40 in a 2x2 grid.
"""
import os, sys, json, glob, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['mlp1', 'firstlayer'], default='mlp1',
                    help='Which L1 hijack to show: mlp1 (default) or firstlayer')
ARGS = parser.parse_args()

DATADIR = os.path.join(os.path.dirname(__file__),
                       'data_allI' if ARGS.mode == 'mlp1' else 'data_allI_v2')
OUTDIR  = os.path.join(os.path.dirname(__file__), 'plots')
os.makedirs(OUTDIR, exist_ok=True)

GAPS = [1, 5, 10, 20, 40, 60]
SEEDS = ['seed1', 'seed2', 'seed3', 'seed4', 'seed5']

if ARGS.mode == 'firstlayer':
    HT_STYLES = {
        'firstlayer': {'color': '#e41a1c', 'label': 'ATTN1 direct circuit hijack',  'ls': '-'},
        'attn2':      {'color': '#2166ac', 'label': 'ATTN2 hijack',               'ls': '-'},
        'and_fl':     {'color': '#984ea3', 'label': 'ATTN1 direct+ATTN2 individually succeed', 'ls': '--'},
    }
else:
    HT_STYLES = {
        'mlp1':  {'color': '#d6604d', 'label': 'MLP1 hijack',              'ls': '-'},
        'attn2': {'color': '#2166ac', 'label': 'ATTN2 hijack',             'ls': '-'},
        'and':   {'color': '#984ea3', 'label': 'Both individually succeed', 'ls': '--'},
    }

plt.rcParams.update({
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'savefig.facecolor': 'white', 'axes.spines.top': False,
    'axes.spines.right': False, 'font.size': 10,
})


def load_all_data():
    """Load JSON files -> {gap: {seed: data_dict}}"""
    all_data = {}
    for gap in GAPS:
        all_data[gap] = {}
        for seed in SEEDS:
            path = os.path.join(DATADIR, f'{seed}_gap{gap}.json')
            if os.path.exists(path):
                with open(path) as f:
                    all_data[gap][seed] = json.load(f)
            else:
                print(f"  WARNING: missing {path}")
    return all_data


def plot_combined(all_data):
    fig, axes = plt.subplots(2, 3, figsize=(20, 9))
    axes = axes.flatten()

    for idx, gap in enumerate(GAPS):
        ax = axes[idx]
        seed_data = all_data[gap]
        n_seeds = len(seed_data)
        if n_seeds == 0:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center')
            continue

        for ht, style in HT_STYLES.items():
            ref = list(seed_data.values())[0]
            if ht == 'and_fl':
                src_key = 'firstlayer'
            elif ht == 'and':
                src_key = 'mlp1'
            else:
                src_key = ht
            if src_key not in ref:
                continue
            offsets = ref[src_key]['offsets']
            if not offsets:
                continue

            seed_rates = []
            for seed in SEEDS:
                if seed not in seed_data:
                    continue
                sd = seed_data[seed]
                if ht in ('and_fl', 'and'):
                    s_offsets = sd[src_key]['offsets']
                    s_rates_x = sd[src_key]['rates']
                    s_rates_a = sd['attn2']['rates']
                    a_offsets = sd['attn2']['offsets']
                    a_map = dict(zip(a_offsets, s_rates_a))
                    row = [min(r, a_map.get(o, 0)) for o, r in zip(s_offsets, s_rates_x)]
                    off_to_rate = dict(zip(s_offsets, row))
                else:
                    s_offsets = sd[ht]['offsets']
                    s_rates = sd[ht]['rates']
                    off_to_rate = dict(zip(s_offsets, s_rates))
                row = [off_to_rate.get(o, np.nan) for o in offsets]
                seed_rates.append(row)

            seed_rates = np.array(seed_rates)
            mean = np.nanmean(seed_rates, axis=0)
            std = np.nanstd(seed_rates, axis=0)

            marker = 'o' if style['ls'] == '-' else 's'
            ax.plot(offsets, mean, linestyle=style['ls'], marker=marker,
                    color=style['color'], linewidth=2, markersize=4,
                    alpha=0.9, label=style['label'])
            ax.fill_between(offsets, mean - std, mean + std,
                            color=style['color'], alpha=0.15)

            for seed_name, row in zip([s for s in SEEDS if s in seed_data], seed_rates):
                ax.plot(offsets, row, linestyle=style['ls'],
                        color=style['color'], linewidth=0.5, alpha=0.25)

        ax.set_ylim(-5, 105)
        ax.set_title(f'gap = {gap}', fontsize=13, fontweight='bold')
        ax.set_xlabel('Offset (hijack to $i$ + offset)', fontsize=10)
        if idx % 3 == 0:
            ax.set_ylabel('Hijack success rate (%)', fontsize=10)
        ax.text(0.98, 0.02, f'{n_seeds} seeds', transform=ax.transAxes,
                fontsize=8, ha='right', va='bottom', color='gray')

    for ax in axes:
        ax.legend(fontsize=8, loc='best', framealpha=0.9)
    fig.suptitle('Hijack success rate averaged over all $i$ and all seeds (k32_N512)',
                 fontsize=14, fontweight='bold', y=1.01)
    fig.tight_layout()
    tag = '_fl' if ARGS.mode == 'firstlayer' else ''
    out_path = os.path.join(OUTDIR, f'hijack_allI_avg_seeds{tag}.png')
    fig.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved {out_path}")


if __name__ == '__main__':
    print("Loading data ...")
    all_data = load_all_data()
    for gap in GAPS:
        print(f"  gap={gap}: {len(all_data[gap])} seeds loaded")
    plot_combined(all_data)
    print("Done.")
