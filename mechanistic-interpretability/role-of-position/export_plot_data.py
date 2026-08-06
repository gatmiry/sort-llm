#!/usr/bin/env python3
"""
Export the curves drawn by plot_hijack_avg_seeds.py to a single CSV.

The figure averages 23 per-seed JSON files per panel, so no single file in
data_allI_v3 holds a plotted curve. This writes the cross-seed mean and standard
deviation actually rendered, one row per (gap, curve, offset).

The aggregation mirrors plot_combined() in plot_hijack_avg_seeds.py: the offset
axis comes from the first loaded seed, other seeds are looked up by exact offset
value and contribute NaN when absent, and the joint curve is the per-offset
minimum of the two individual rates.

Usage:
  python export_plot_data.py --datadir data_allI_v3 \
      --classification leapformer_classification.json --out hijack_v3_curves.csv
"""
import argparse
import csv
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GAPS = [1, 5, 10, 20, 40, 60]
ALL_SEEDS = list(range(1, 26))

CURVE_LABELS = {
    'firstlayer': 'ATTN1 direct circuit hijack',
    'attn2': 'ATTN2 hijack',
    'and_fl': 'ATTN1 direct+ATTN2 individually succeed',
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', default=os.path.join(HERE, 'data_allI_v3'))
    parser.add_argument('--classification',
                        default=os.path.join(HERE, 'leapformer_classification.json'))
    parser.add_argument('--exclude-seeds', default='8,10',
                        help='Used only when --classification is absent')
    parser.add_argument('--out', default=os.path.join(HERE, 'hijack_v3_curves.csv'))
    args = parser.parse_args()

    if os.path.exists(args.classification):
        cls = json.load(open(args.classification))['seeds']
        excluded = sorted(int(s) for s, r in cls.items() if not r['is_leap'])
    else:
        excluded = sorted(int(s) for s in args.exclude_seeds.split(',') if s.strip())
    seeds = [f'seed{i}' for i in ALL_SEEDS if i not in excluded]
    print(f'Excluding seeds {excluded}; averaging over {len(seeds)}')

    rows = []
    for gap in GAPS:
        data = {}
        for seed in seeds:
            path = os.path.join(args.datadir, f'{seed}_gap{gap}.json')
            if os.path.exists(path):
                data[seed] = json.load(open(path))
            else:
                print(f'  WARNING: missing {path}')
        if not data:
            continue

        for curve in ('firstlayer', 'attn2', 'and_fl'):
            src = 'firstlayer' if curve == 'and_fl' else curve
            offsets = list(data.values())[0][src]['offsets']

            per_seed = []
            for seed in seeds:
                if seed not in data:
                    continue
                sd = data[seed]
                if curve == 'and_fl':
                    a_map = dict(zip(sd['attn2']['offsets'], sd['attn2']['rates']))
                    pairs = zip(sd[src]['offsets'], sd[src]['rates'])
                    off_to_rate = {o: min(r, a_map.get(o, 0)) for o, r in pairs}
                else:
                    off_to_rate = dict(zip(sd[curve]['offsets'], sd[curve]['rates']))
                per_seed.append([off_to_rate.get(o, np.nan) for o in offsets])

            arr = np.array(per_seed)
            mean = np.nanmean(arr, axis=0)
            std = np.nanstd(arr, axis=0)
            contributing = np.sum(~np.isnan(arr), axis=0)
            for i, off in enumerate(offsets):
                rows.append({
                    'gap': gap,
                    'offset': off,
                    'curve': curve,
                    'curve_label': CURVE_LABELS[curve],
                    'mean_pct': round(float(mean[i]), 4),
                    'std_pct': round(float(std[i]), 4),
                    'n_seeds': int(contributing[i]),
                })

    with open(args.out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Saved {args.out} ({len(rows)} rows)')


if __name__ == '__main__':
    main()
