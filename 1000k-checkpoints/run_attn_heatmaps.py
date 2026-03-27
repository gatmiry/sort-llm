#!/usr/bin/env python3
"""
Run attn_heatmaps2.py for all checkpoints in 1000k-checkpoints/,
saving to the matching outputs/plots_V256_B16_.../ subfolder.
Distributes across available GPUs.
"""
import os
import glob
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HEATMAP_SCRIPT = os.path.join(SCRIPT_DIR, '..', 'heat-map-code', 'attn_heatmaps2.py')
OUTPUT_BASE = os.path.join(SCRIPT_DIR, 'outputs')


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
    pt_files = sorted(glob.glob(os.path.join(SCRIPT_DIR, '*.pt')))
    if not pt_files:
        print("No .pt files found!")
        sys.exit(1)

    import torch
    num_gpus = torch.cuda.device_count() or 1

    tasks = []
    for pt in pt_files:
        bn = os.path.basename(pt)
        itr = get_iter_from_filename(bn)
        if itr is None:
            print(f"Skipping unrecognised file: {bn}")
            continue
        folder = make_folder_name(itr)
        out_dir = os.path.join(OUTPUT_BASE, folder)
        os.makedirs(out_dir, exist_ok=True)

        dest = os.path.join(out_dir, 'attn_heatmaps.png')
        if os.path.exists(dest):
            print(f"Already exists, skipping: {dest}")
            continue
        tasks.append((pt, out_dir))

    if not tasks:
        print("All heatmaps already generated.")
        return

    print(f"Generating heatmaps for {len(tasks)} checkpoints on {num_gpus} GPUs ...")

    procs = []
    for idx, (pt_path, out_dir) in enumerate(tasks):
        gpu_id = idx % num_gpus
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        cmd = [sys.executable, HEATMAP_SCRIPT, pt_path, '--output-dir', out_dir]
        print(f"  [{gpu_id}] {os.path.basename(pt_path)} -> {os.path.basename(out_dir)}")
        p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        procs.append((p, pt_path, out_dir))

        if len(procs) >= num_gpus:
            for p, pt, od in procs:
                stdout, stderr = p.communicate()
                if p.returncode != 0:
                    print(f"  FAILED: {os.path.basename(pt)}\n{stderr.decode()}")
                else:
                    print(f"  Done: {os.path.basename(pt)}")
            procs = []

    for p, pt, od in procs:
        stdout, stderr = p.communicate()
        if p.returncode != 0:
            print(f"  FAILED: {os.path.basename(pt)}\n{stderr.decode()}")
        else:
            print(f"  Done: {os.path.basename(pt)}")

    print("All done.")


if __name__ == '__main__':
    main()
