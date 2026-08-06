#!/usr/bin/env python3
"""
Ray driver for Step 5 of HANDOFF_finegrained_hijack.md: label each seed
leap-former or single-stage.

classify_new_seeds.py needs a GPU and the checkpoints, both of which live on the
Ray worker, so it is run there and the resulting JSON is written back on the
head. The criterion (per-token accuracy with attn2 ablated below 10%) and the
JSON layout come from classify_new_seeds.py unchanged.

Usage:
  python classify_seeds_ray.py --out leapformer_classification.json
"""
import argparse
import json
import os
import subprocess
import sys

import ray

from ray_stage import HERE, REL_CLASSIFY, build_stage, parse_seeds


@ray.remote(num_gpus=1, num_cpus=4)
def run_classify(seeds_spec):
    out_path = '/tmp/leapformer_classification.json'
    env = dict(os.environ)
    env['MPLCONFIGDIR'] = '/tmp/mplconfig'
    proc = subprocess.run(
        [sys.executable, REL_CLASSIFY, '--seeds', seeds_spec, '--save-json', out_path],
        capture_output=True, text=True, env=env,
    )
    payload = None
    if proc.returncode == 0 and os.path.exists(out_path):
        with open(out_path) as f:
            payload = json.load(f)
    return payload, proc.stdout + proc.stderr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', default='1-25')
    parser.add_argument('--out', default=os.path.join(HERE, 'leapformer_classification.json'))
    parser.add_argument('--stage-dir', default='/tmp/classify_stage')
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    stage = build_stage(seeds, args.stage_dir)
    ray.init(address='auto', runtime_env={'working_dir': stage})

    payload, log = ray.get(run_classify.remote(args.seeds))
    print(log)
    if payload is None:
        raise SystemExit('classification failed, see log above')

    with open(args.out, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f'Saved {args.out}')

    singles = sorted(int(s) for s, r in payload['seeds'].items() if not r['is_leap'])
    leaps = sorted(int(s) for s, r in payload['seeds'].items() if r['is_leap'])
    print(f'Leap-formers ({len(leaps)}): {leaps}')
    print(f'Single-stage ({len(singles)}): {singles}')
    if singles != [8, 10]:
        # Section 8 of the handoff: a different set is a real finding about the
        # models, not a misconfiguration, and it changes the paper's numbers.
        print(f'WARNING: expected single-stage seeds [8, 10], got {singles}')


if __name__ == '__main__':
    main()
