#!/usr/bin/env python3
"""
Ray driver for the fine-grained hijack sweep (Steps 2-3 of
HANDOFF_finegrained_hijack.md).

Same job list and same per-job command as run_finegrained_seeds.sh -- the offset
chunk table is parsed out of that script so the two cannot drift -- but the jobs
are dispatched to a Ray cluster instead of to local GPUs.

The GPU node does not share a filesystem with the head, so each task writes its
JSON to the worker's local disk and returns the parsed contents, which the head
writes into --datadir. Outputs appear one at a time and existing ones are
skipped, so the sweep is resumable exactly like the bash runner.

Each job is one batch-size-1 python process that spends most of its time on
kernel launch latency rather than on the GPU, so several are packed onto every
GPU via fractional num_gpus.

Usage:
  python run_finegrained_ray.py                      # full sweep
  python run_finegrained_ray.py --seeds 1 --gaps 60  # one seed, one gap
  python run_finegrained_ray.py --provenance         # Step 2 check only
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

import ray

from ray_stage import HERE, REL_SWEEP, build_stage, checkpoint_rel, parse_seeds

BASH_RUNNER = os.path.join(HERE, 'run_finegrained_seeds.sh')
WORKER_LOG_DIR = '/tmp/hijack_job_logs'


def parse_bash_config():
    """Read GAPS and the CHUNKS table out of run_finegrained_seeds.sh."""
    with open(BASH_RUNNER) as f:
        text = f.read()
    gaps_m = re.search(r'^GAPS=\(([^)]*)\)', text, re.M)
    if not gaps_m:
        raise SystemExit(f'no GAPS=(...) line in {BASH_RUNNER}')
    gaps = [int(x) for x in gaps_m.group(1).split()]
    chunks = {}
    for m in re.finditer(r'^CHUNKS\[(\d+)\]="([^"]*)"', text, re.M):
        chunks[int(m.group(1))] = m.group(2).split()
    missing = [g for g in gaps if g not in chunks]
    if missing:
        raise SystemExit(f'no CHUNKS entry for gaps {missing} in {BASH_RUNNER}')
    return gaps, chunks


@ray.remote
def run_job(gap, offsets, ckpt_rel, out_tag, max_batches):
    """Run one plot_hijack_per_i.py invocation on the worker.

    Returns (payload, elapsed_seconds, log); payload is None on failure.
    """
    out_path = f'/tmp/hijack_out_{out_tag}.json'
    env = dict(os.environ)
    # Jobs are packed several to a GPU; without this each torch process grabs a
    # large intraop thread pool and they trample each other on the CPU side.
    env['OMP_NUM_THREADS'] = '1'
    env['MPLCONFIGDIR'] = '/tmp/mplconfig'

    cmd = [
        sys.executable, REL_SWEEP,
        '--gap', str(gap),
        '--offsets', offsets,
        '--group-avg', '0-497',
        '--ckpt', ckpt_rel,
        '--out-tag', out_tag,
        '--max-batches', str(max_batches),
        '--save-data', out_path,
    ]
    # Stream to a file on the worker rather than a pipe: a job runs for tens of
    # minutes and its progress lines are the only way to see how far along it is
    # before it returns.
    os.makedirs(WORKER_LOG_DIR, exist_ok=True)
    log_path = os.path.join(WORKER_LOG_DIR, f'{out_tag}.log')
    t0 = time.time()
    with open(log_path, 'w') as log_file:
        proc = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT, env=env)
    elapsed = time.time() - t0

    with open(log_path) as f:
        log = f.read()

    payload = None
    if proc.returncode == 0 and os.path.exists(out_path):
        with open(out_path) as f:
            payload = json.load(f)
        os.remove(out_path)
    return payload, elapsed, log


def submit(jobs, datadir, jobs_per_gpu, cpus_per_job):
    os.makedirs(datadir, exist_ok=True)
    os.makedirs(os.path.join(datadir, 'logs'), exist_ok=True)

    pending = {}
    for out_name, gap, offsets, ckpt_rel, tag, mb in jobs:
        ref = run_job.options(
            num_gpus=1.0 / jobs_per_gpu,
            num_cpus=cpus_per_job,
        ).remote(gap, offsets, ckpt_rel, tag, mb)
        pending[ref] = out_name

    total = len(pending)
    done = failed = 0
    t0 = time.time()
    while pending:
        ready, _ = ray.wait(list(pending), num_returns=1)
        ref = ready[0]
        out_name = pending.pop(ref)
        try:
            payload, elapsed, log = ray.get(ref)
        except Exception as exc:
            failed += 1
            print(f'  FAILED {out_name}: {exc}', flush=True)
            continue

        with open(os.path.join(datadir, 'logs', out_name.replace('.json', '.log')), 'w') as f:
            f.write(log)

        if payload is None or not payload.get('attn2', {}).get('rates'):
            failed += 1
            reason = 'no output' if payload is None else 'empty attn2 rates'
            print(f'  FAILED {out_name} ({reason}), see logs/', flush=True)
            continue

        with open(os.path.join(datadir, out_name), 'w') as f:
            json.dump(payload, f, indent=2)
        done += 1
        wall = time.time() - t0
        eta = (total - done - failed) * wall / done / 60 if done else float('inf')
        print(f'  [{done + failed}/{total}] {out_name} '
              f'{len(payload["attn2"]["rates"])} offsets, n={payload["n_total"]}, '
              f'{elapsed / 60:.1f} min | eta {eta:.0f} min', flush=True)

    return done, failed


def build_jobs(seeds, gaps, chunks, datadir, mb_random, mb_targeted):
    jobs = []
    for seed in seeds:
        ckpt_rel = checkpoint_rel(seed)
        for gap in gaps:
            for chunk in chunks[gap]:
                out_name = f'seed{seed}_gap{gap}__{chunk}.json'
                if os.path.exists(os.path.join(datadir, out_name)):
                    continue
                lo, hi = (int(x) for x in chunk.split('-'))
                offsets = ','.join(str(o) for o in range(lo, hi + 1))
                mb = mb_targeted if gap >= 20 else mb_random
                jobs.append((out_name, gap, offsets, ckpt_rel,
                             f'seed{seed}_allI_v3_{chunk}', mb))
    return jobs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', default=os.path.join(HERE, 'data_allI_v3'))
    parser.add_argument('--stage-dir', default='/tmp/hijack_stage')
    parser.add_argument('--seeds', default='1-25')
    parser.add_argument('--gaps', default=None, help='Comma-separated subset of gaps')
    parser.add_argument('--jobs-per-gpu', type=int, default=6)
    parser.add_argument('--cpus-per-job', type=int, default=2)
    parser.add_argument('--mb-random', type=int, default=60000)
    parser.add_argument('--mb-targeted', type=int, default=40000)
    parser.add_argument('--provenance', action='store_true',
                        help='Run only the seed 1 gap 60 reproduction of data_allI_v2')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    gaps, chunks = parse_bash_config()
    if args.gaps:
        wanted = {int(g) for g in args.gaps.split(',')}
        gaps = [g for g in gaps if g in wanted]

    datadir = args.datadir
    if args.provenance:
        # Reproduces one cell that already exists in data_allI_v2 so the
        # Hugging Face checkpoints can be compared against the committed rates.
        seeds = [1]
        datadir = os.path.join(datadir, 'provenance')
        jobs = [('seed1_gap60__provenance.json', 60,
                 '61,65,70,80,90,100,120,150', checkpoint_rel(1),
                 'provenance_check', 5000)]
    else:
        jobs = build_jobs(seeds, gaps, chunks, datadir,
                          args.mb_random, args.mb_targeted)

    if not jobs:
        print('Nothing to do (all outputs already exist).')
        return

    print(f'{len(jobs)} jobs, {args.jobs_per_gpu} per GPU -> {datadir}', flush=True)
    if args.dry_run:
        for out_name, gap, _, _, _, mb in jobs[:10]:
            print(f'   {out_name} gap={gap} mb={mb}')
        print(f'   ... {len(jobs)} total')
        return

    stage = build_stage(seeds, args.stage_dir)
    ray.init(address='auto', runtime_env={'working_dir': stage})
    print(ray.cluster_resources(), flush=True)

    t0 = time.time()
    done, failed = submit(jobs, datadir, args.jobs_per_gpu, args.cpus_per_job)
    print(f'\n{done} written, {failed} failed, {(time.time() - t0) / 60:.1f} min total')
    if not args.provenance:
        print(f'Next: python merge_hijack_chunks.py --datadir {datadir}')


if __name__ == '__main__':
    main()
