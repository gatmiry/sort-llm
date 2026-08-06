#!/usr/bin/env python3
"""
Merge chunked hijack sweeps written by run_finegrained_seeds.sh.

Reads {seed}_gap{gap}__{lo}-{hi}.json and writes {seed}_gap{gap}.json in the
format plot_hijack_avg_seeds.py expects.

Also audits the result. plot_hijack_avg_seeds.py takes its x-axis from the
first seed and looks every other seed up by exact offset value, so a seed swept
on a different grid contributes nothing while still being counted in the
"N seeds" annotation. Any grid mismatch reported here would silently shrink the
averages in the figure.
"""
import os, re, json, glob, argparse, collections

HT_KEYS = ['mlp1', 'attn2', 'both', 'and', 'firstlayer']
CHUNK_RE = re.compile(r'^(seed\d+)_gap(\d+)__(\d+)-(\d+)\.json$')

parser = argparse.ArgumentParser()
parser.add_argument('--datadir', default=os.path.join(os.path.dirname(__file__), 'data_allI_v3'))
parser.add_argument('--out-dir', default=None, help='Defaults to --datadir')
parser.add_argument('--dry-run', action='store_true')
ARGS = parser.parse_args()

OUTDIR = ARGS.out_dir or ARGS.datadir


def collect_chunks():
    """{(seed, gap): [chunk_path, ...]}"""
    groups = collections.defaultdict(list)
    for path in sorted(glob.glob(os.path.join(ARGS.datadir, 'seed*_gap*__*.json'))):
        m = CHUNK_RE.match(os.path.basename(path))
        if m:
            groups[(m.group(1), int(m.group(2)))].append(path)
    return groups


def merge_one(paths):
    merged = {ht: {} for ht in HT_KEYS}
    requested, n_total, conflicts, empty = set(), 0, [], []

    for path in paths:
        with open(path) as f:
            data = json.load(f)
        n_total += data.get('n_total', 0)
        requested.update(data.get('offsets', []))
        if not data.get('attn2', {}).get('rates'):
            empty.append(os.path.basename(path))
        for ht in HT_KEYS:
            block = data.get(ht, {})
            for off, rate in zip(block.get('offsets', []), block.get('rates', [])):
                if off in merged[ht] and abs(merged[ht][off] - rate) > 1e-9:
                    conflicts.append((ht, off))
                merged[ht][off] = rate

    out = {'offsets': sorted(requested), 'n_total': n_total}
    for ht in HT_KEYS:
        offs = sorted(merged[ht])
        out[ht] = {'offsets': offs, 'rates': [merged[ht][o] for o in offs]}
    return out, empty, conflicts


def main():
    groups = collect_chunks()
    if not groups:
        print(f"No chunk files found in {ARGS.datadir}")
        return

    grids = collections.defaultdict(dict)
    for (seed, gap), paths in sorted(groups.items(), key=lambda kv: (kv[0][1], int(kv[0][0][4:]))):
        out, empty, conflicts = merge_one(paths)
        out['gap'] = gap
        grids[gap][seed] = tuple(out['firstlayer']['offsets'])

        dest = os.path.join(OUTDIR, f'{seed}_gap{gap}.json')
        n_off = len(out['firstlayer']['offsets'])
        print(f'{seed:>8s} gap{gap:<3d} {len(paths)} chunks -> {n_off:3d} offsets, n_total={out["n_total"]}')
        for name in empty:
            print(f'           WARNING: empty chunk {name}')
        if conflicts:
            print(f'           WARNING: {len(conflicts)} duplicated offsets across chunks')
        if not ARGS.dry_run:
            os.makedirs(OUTDIR, exist_ok=True)
            with open(dest, 'w') as f:
                json.dump(out, f, indent=2)

    print('\nGrid audit (all seeds within a gap must share one grid):')
    for gap in sorted(grids):
        distinct = collections.defaultdict(list)
        for seed, grid in grids[gap].items():
            distinct[grid].append(seed)
        if len(distinct) == 1 and all(distinct):
            grid = next(iter(distinct))
            print(f'  gap{gap:<3d} OK - {len(grids[gap])} seeds on {len(grid)} shared offsets')
        else:
            print(f'  gap{gap:<3d} MISMATCH - {len(distinct)} distinct grids:')
            for grid, seeds in distinct.items():
                label = f'{len(grid)} offsets' if grid else 'EMPTY'
                print(f'          {label:>12s}: {seeds}')


if __name__ == '__main__':
    main()
