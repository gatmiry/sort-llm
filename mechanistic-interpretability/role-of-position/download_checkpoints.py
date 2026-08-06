#!/usr/bin/env python3
"""Fetch the 25 k32_N512 100k-step checkpoints from Hugging Face into the local
trees the role-of-position scripts expect (Step 2 of
HANDOFF_finegrained_hijack.md).

Checkpoints are gitignored (`*.pt`), so this is the way to populate a fresh
clone. About 0.7 MB each, ~17 MB total. Already-present files are left alone.
"""
import os
import shutil

from huggingface_hub import hf_hub_download

from ray_stage import REPO_ROOT, checkpoint_rel

HF_REPO = 'gatmiry/sortgpt-checkpoints'


def main():
    for seed in range(1, 26):
        rel = checkpoint_rel(seed)
        out_path = os.path.join(REPO_ROOT, rel)
        if os.path.exists(out_path):
            print('have', rel, flush=True)
            continue
        fn = os.path.basename(rel)
        src = hf_hub_download(HF_REPO, f'checkpoints/k32_N512/seed{seed}/{fn}')
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        shutil.copy(src, out_path)
        print('ok', rel, flush=True)

    have = sum(os.path.exists(os.path.join(REPO_ROOT, checkpoint_rel(s)))
               for s in range(1, 26))
    print(f'\n{have}/25 checkpoints present')


if __name__ == '__main__':
    main()
