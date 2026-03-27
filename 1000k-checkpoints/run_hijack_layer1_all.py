#!/usr/bin/env python3
"""
Run hijack intervention on layer 1 for ALL checkpoints in 1000k-checkpoints/,
distributed across available GPUs. Saves heatmaps in the matching output subfolder.
"""
import os
import sys
import glob
import json
import subprocess
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_BASE = os.path.join(SCRIPT_DIR, 'outputs')
WORKER_SCRIPT = os.path.join(SCRIPT_DIR, 'hijack_layer1_worker.py')


def get_iter_from_filename(basename):
    parts = basename.replace('.pt', '').split('__')
    ckpt_type = parts[1] if len(parts) > 1 else 'final'
    if ckpt_type.startswith('ckpt'):
        return int(ckpt_type.replace('ckpt', ''))
    elif ckpt_type == 'final':
        return 1000000
    return None


def make_folder_name(itr):
    return f"plots_V256_B16_LR3e-2_MI{itr}_E64_H1_L2_ds1337_is1337_ckpt{itr}"


def main():
    import torch
    num_gpus = torch.cuda.device_count() or 1

    pt_files = sorted(glob.glob(os.path.join(SCRIPT_DIR, '*.pt')))
    tasks = []
    for pt in pt_files:
        bn = os.path.basename(pt)
        itr = get_iter_from_filename(bn)
        if itr is None:
            continue
        folder = make_folder_name(itr)
        out_dir = os.path.join(OUTPUT_BASE, folder)
        check_file = os.path.join(out_dir, 'hijack_breaking_rate_heatmap_layer1.png')
        if os.path.exists(check_file):
            print(f"Already exists, skipping: {bn}")
            continue
        tasks.append((pt, out_dir, itr))

    if not tasks:
        print("All layer-1 hijack heatmaps already generated.")
        return

    print(f"Running layer-1 hijack for {len(tasks)} checkpoints on {num_gpus} GPUs ...\n")

    batch_size = num_gpus
    for batch_start in range(0, len(tasks), batch_size):
        batch = tasks[batch_start:batch_start + batch_size]
        procs = []
        for idx, (pt_path, out_dir, itr) in enumerate(batch):
            gpu_id = idx % num_gpus
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            cmd = [sys.executable, WORKER_SCRIPT, pt_path, '--output-dir', out_dir]
            bn = os.path.basename(pt_path)
            print(f"  [GPU {gpu_id}] {bn} -> {os.path.basename(out_dir)}")
            p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            procs.append((p, bn))

        for p, bn in procs:
            stdout, stderr = p.communicate()
            out_text = stdout.decode().strip()
            if p.returncode != 0:
                print(f"  FAILED: {bn}\n{stderr.decode()[-500:]}")
            else:
                last_lines = [l for l in out_text.split('\n') if l.strip()][-3:]
                for l in last_lines:
                    print(f"    {l}")
                print(f"  Done: {bn}")
        print()

    print("All done.")


if __name__ == '__main__':
    main()
