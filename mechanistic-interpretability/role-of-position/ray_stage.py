"""Shared staging helpers for the Ray drivers in this directory.

The GPU node does not share a filesystem with the Ray head, so work is shipped
as a runtime_env working_dir. The repo is 17 GB, almost all of it checkpoints
and figures the hijack scripts never open, so the staging tree mirrors only the
handful of files a job actually touches (~18 MB).
"""
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))

REL_SWEEP = 'mechanistic-interpretability/role-of-position/plot_hijack_per_i.py'
REL_CLASSIFY = 'mechanistic-interpretability/role-of-position/classify_new_seeds.py'


def checkpoint_rel(seed):
    """Repo-relative path of a seed's 100k checkpoint.

    Mirrors checkpoint_for() in classify_new_seeds.py and the CKPTS table in
    run_finegrained_seeds.sh; the four trees are a historical artefact of the
    seeds having been trained in four separate batches.
    """
    if seed == 1:
        d = 'new-grid/k32_N512/checkpoints'
    elif seed <= 5:
        d = f'new-grid-multiple/k32_N512/seed{seed}/checkpoints'
    elif seed <= 15:
        d = f'new-grid-multiple-2/k32_N512/seed{seed}/checkpoints'
    else:
        d = f'new-grid-multiple-3/k32_N512/seed{seed}/checkpoints'
    return f'{d}/std0p01_iseed{seed}__ckpt100000.pt'


def build_stage(seeds, stage_dir):
    """Mirror the repo layout with only the files a GPU job needs.

    plot_hijack_per_i.py and classify_new_seeds.py both reach the toolkit via a
    path relative to their own location, so the nesting must be preserved.
    """
    if os.path.isdir(stage_dir):
        shutil.rmtree(stage_dir)
    os.makedirs(stage_dir)

    dst_toolkit = os.path.join(stage_dir, 'sortgpt_toolkit')
    os.makedirs(dst_toolkit)
    for name in ('model.py', 'intervene.py'):
        shutil.copy(os.path.join(REPO_ROOT, 'sortgpt_toolkit', name), dst_toolkit)

    dst_rop = os.path.join(stage_dir, 'mechanistic-interpretability', 'role-of-position')
    os.makedirs(dst_rop)
    for name in ('plot_hijack_per_i.py', 'classify_new_seeds.py'):
        shutil.copy(os.path.join(HERE, name), dst_rop)

    missing = []
    for seed in seeds:
        rel = checkpoint_rel(seed)
        src = os.path.join(REPO_ROOT, rel)
        if not os.path.exists(src):
            missing.append(rel)
            continue
        dst = os.path.join(stage_dir, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(src, dst)
    if missing:
        raise SystemExit('missing checkpoints:\n  ' + '\n  '.join(missing))

    size_mb = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fs in os.walk(stage_dir) for f in fs
    ) / 1e6
    print(f'staged {stage_dir} ({size_mb:.1f} MB)')
    return stage_dir


def parse_seeds(spec):
    out = []
    for part in spec.split(','):
        if '-' in part:
            lo, hi = (int(x) for x in part.split('-'))
            out.extend(range(lo, hi + 1))
        else:
            out.append(int(part))
    return sorted(set(out))
