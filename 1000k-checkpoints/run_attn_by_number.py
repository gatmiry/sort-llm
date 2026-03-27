#!/usr/bin/env python3
"""
Orchestrator: distribute avg-attention-by-number computation across 8 GPUs
for all checkpoints in 1000k-checkpoints/.
"""
import glob, json, os, subprocess, sys, time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKER = os.path.join(SCRIPT_DIR, 'attn_by_number_worker.py')
OUTPUT_BASE = os.path.join(SCRIPT_DIR, 'outputs')
TASK_DIR = os.path.join(OUTPUT_BASE, 'task_files')


def get_iter(basename):
    tag = basename.replace('.pt', '').split('__')[1]
    if tag.startswith('ckpt'):
        return int(tag.replace('ckpt', ''))
    if tag == 'final':
        return 1000000
    return None


def folder_name(itr):
    return f"plots_V256_B16_LR3e-2_MI{itr}_E64_H1_L2_ds1337_is1337_ckpt{itr}"


def main():
    import torch
    n_gpus = min(torch.cuda.device_count(), 8)

    pt_files = sorted(glob.glob(os.path.join(SCRIPT_DIR, '*.pt')))
    seen, tasks = set(), []
    for pt in pt_files:
        bn = os.path.basename(pt)
        itr = get_iter(bn)
        if itr is None or itr in seen:
            continue
        seen.add(itr)
        tasks.append({'ckpt_path': pt,
                      'out_dir': os.path.join(OUTPUT_BASE, folder_name(itr)),
                      'itr': itr})
    tasks.sort(key=lambda t: t['itr'])

    print(f"{len(tasks)} checkpoints → {n_gpus} GPUs")

    gpu_tasks = [[] for _ in range(n_gpus)]
    for i, t in enumerate(tasks):
        gpu_tasks[i % n_gpus].append(t)

    os.makedirs(TASK_DIR, exist_ok=True)
    procs = []
    for gid in range(n_gpus):
        if not gpu_tasks[gid]:
            continue
        tf = os.path.join(TASK_DIR, f'attn_by_number_gpu{gid}.json')
        with open(tf, 'w') as f:
            json.dump(gpu_tasks[gid], f)
        labels = [f"ckpt{t['itr']//1000}k" for t in gpu_tasks[gid]]
        print(f"  GPU {gid}: {', '.join(labels)}")
        p = subprocess.Popen(
            [sys.executable, WORKER, '--tasks-file', tf, '--gpu', str(gid)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        procs.append((p, gid))

    t0 = time.time()
    for p, gid in procs:
        stdout, stderr = p.communicate()
        rc = p.returncode
        tag = "OK" if rc == 0 else f"FAIL(rc={rc})"
        print(f"  GPU {gid}: {tag}")
        if stdout:
            for line in stdout.decode().strip().split('\n'):
                print(f"    {line}")
        if rc != 0 and stderr:
            print(f"    STDERR: {stderr.decode()[-800:]}")

    print(f"\nDone in {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
