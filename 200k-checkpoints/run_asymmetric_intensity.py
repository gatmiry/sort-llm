"""
Run two asymmetric intensity experiments on layer 0:
  1. ub=60, lb=0  (intervene on one number ABOVE target within range 60)
  2. ub=0,  lb=60  (intervene on one number BELOW target within range 60)
Plots both curves on a single figure.
"""
import json
import os
import subprocess
import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_BASE = os.path.join(SCRIPT_DIR, 'outputs')
CKPT = os.path.join(SCRIPT_DIR,
    'sortgpt_k16_methfixed_mlp1_L2_N256_E64_pos0_fln1_wd0p0_lr0p03_dseed1337_iseed1337__final.pt')

FOLDER = 'plots_V256_B16_LR3e-2_MI200000_E64_H1_L2_ds1337_is1337_ckpt200000'
TMP_DIR = os.path.join(OUTPUT_BASE, 'tmp_results', FOLDER)
PLOT_DIR = os.path.join(OUTPUT_BASE, FOLDER)


def main():
    os.makedirs(TMP_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    tasks = [
        {'ckpt_path': CKPT, 'type': 'intensity_asym', 'layer': 0,
         'unsorted_ub': 60, 'unsorted_lb': 0,
         'unsorted_ub_num': 1, 'unsorted_lb_num': 0,
         'out': os.path.join(TMP_DIR, 'intensity_layer0_ub60_lb0.npz'),
         'name': 'intensity_layer0_ub60_lb0', 'itr': 200000},
        {'ckpt_path': CKPT, 'type': 'intensity_asym', 'layer': 0,
         'unsorted_ub': 0, 'unsorted_lb': 60,
         'unsorted_ub_num': 0, 'unsorted_lb_num': 1,
         'out': os.path.join(TMP_DIR, 'intensity_layer0_ub0_lb60.npz'),
         'name': 'intensity_layer0_ub0_lb60', 'itr': 200000},
    ]

    to_run = [t for t in tasks if not os.path.exists(t['out'])]
    print(f"Tasks: {len(tasks)}, cached: {len(tasks)-len(to_run)}, to run: {len(to_run)}")

    if to_run:
        task_dir = os.path.join(OUTPUT_BASE, 'task_files')
        log_dir = os.path.join(OUTPUT_BASE, 'gpu_worker_logs')
        os.makedirs(task_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        procs = []
        for i, task in enumerate(to_run):
            gpu = i
            tf = os.path.join(task_dir, f'asym_gpu{gpu}.json')
            with open(tf, 'w') as f:
                json.dump([task], f)
            lf = open(os.path.join(log_dir, f'asym_gpu{gpu}.log'), 'w')
            proc = subprocess.Popen(
                [sys.executable, os.path.join(SCRIPT_DIR, 'gpu_worker.py'),
                 '--tasks-file', tf, '--gpu', str(gpu)],
                stdout=lf, stderr=subprocess.STDOUT, cwd=SCRIPT_DIR)
            procs.append((proc, lf))

        print(f"Launched {len(procs)} workers on GPUs 0-{len(procs)-1}")
        t0 = time.time()
        for proc, lf in procs:
            proc.wait()
            lf.close()
        print(f"Workers done in {time.time()-t0:.1f}s")

        for proc, _ in procs:
            if proc.returncode != 0:
                print(f"  WARN: worker exited {proc.returncode}")

    d_ub = np.load(os.path.join(TMP_DIR, 'intensity_layer0_ub60_lb0.npz'))
    d_lb = np.load(os.path.join(TMP_DIR, 'intensity_layer0_ub0_lb60.npz'))

    tag = 'V=256  B=16  lr=0.03  iters=200000  dseed=1337  iseed=1337'

    plt.figure(figsize=(5.5, 3.8))
    plt.plot(d_ub['intensities'], d_ub['success_rates'],
             marker='o', linewidth=1.8, markersize=6,
             label='ub=60, lb=0 (above target)', color='#e6850e')
    plt.plot(d_lb['intensities'], d_lb['success_rates'],
             marker='s', linewidth=1.8, markersize=6,
             label='ub=0, lb=60 (below target)', color='#1f77b4')
    plt.xlabel('Intervention Intensity', fontsize=10)
    plt.ylabel('Success Probability', fontsize=10)
    plt.title(f'Asymmetric Intervention Robustness (Layer 0)\n{tag}',
              fontsize=11, fontweight='bold')
    plt.legend(fontsize=9, loc='lower left')
    plt.grid(True, alpha=0.3)
    plt.xticks(d_ub['intensities'], fontsize=9)
    plt.yticks(fontsize=9)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    out_path = os.path.join(PLOT_DIR, 'intensity_layer0_asym_ub60_lb60.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved: {out_path}")


if __name__ == '__main__':
    main()
