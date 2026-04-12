#!/usr/bin/env python3
"""
Orchestrator: launch 8 training runs across 8 GPUs, then run analysis/plotting
on the same GPU after each training completes.

Grid: k in [16, 32] × N in [128, 256, 512, 1024]
Fixed: init_std=0.01, lr=0.03, seed=1, n_embd=64, checkpoint_every=5000
"""

import os
import subprocess
import sys
import time
import re

TOOLKIT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "sort-llm", "sortgpt_toolkit")
TRAIN_SCRIPT = os.path.join(TOOLKIT_DIR, "train.py")
PLOT_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "plot_checkpoint_analysis.py")
OUTPUT_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "new-grid")

INIT_STD = 0.01
LR = 0.03
SEED = 1
N_EMBD = 64
MAX_ITERS = 100_000
CHECKPOINT_EVERY = 5000

CONFIGS = [
    {"k": 16, "N": 128,  "gpu": 0},
    {"k": 16, "N": 256,  "gpu": 1},
    {"k": 16, "N": 512,  "gpu": 2},
    {"k": 16, "N": 1024, "gpu": 3},
    {"k": 32, "N": 128,  "gpu": 4},
    {"k": 32, "N": 256,  "gpu": 5},
    {"k": 32, "N": 512,  "gpu": 6},
    {"k": 32, "N": 1024, "gpu": 7},
]


def config_dir_name(cfg):
    return f"k{cfg['k']}_N{cfg['N']}"


def last_checkpoint_path(cfg):
    """Path to the final checkpoint for a config."""
    d = os.path.join(OUTPUT_BASE, config_dir_name(cfg), "checkpoints")
    tag = f"std0p01_iseed{SEED}__ckpt{MAX_ITERS}.pt"
    return os.path.join(d, tag)


def launch_training(cfg):
    run_dir = os.path.join(OUTPUT_BASE, config_dir_name(cfg))
    log_dir = os.path.join(run_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "train.log")

    cmd = [
        sys.executable, TRAIN_SCRIPT,
        "--init-seed", str(SEED),
        "--init-std", str(INIT_STD),
        "--run-dir", run_dir,
        "--max-iters", str(MAX_ITERS),
        "--checkpoint-every", str(CHECKPOINT_EVERY),
        "--lr", str(LR),
        "--block-size", str(cfg["k"]),
        "--vocab-n", str(cfg["N"]),
        "--n-embd", str(N_EMBD),
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(cfg["gpu"])

    log_file = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT,
                            env=env, cwd=TOOLKIT_DIR)
    return proc, log_file, log_path


def launch_plotting(cfg):
    run_dir = os.path.join(OUTPUT_BASE, config_dir_name(cfg))
    plot_dir = os.path.join(run_dir, "plots")
    log_dir = os.path.join(run_dir, "logs")
    os.makedirs(plot_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "plot.log")

    ckpt_path = last_checkpoint_path(cfg)
    tag = f"k={cfg['k']}  N={cfg['N']}  lr={LR}  std={INIT_STD}"

    cmd = [
        sys.executable, PLOT_SCRIPT,
        "--ckpt", ckpt_path,
        "--output-dir", plot_dir,
        "--tag", tag,
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(cfg["gpu"])

    log_file = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT,
                            env=env)
    return proc, log_file, log_path


def main():
    os.makedirs(OUTPUT_BASE, exist_ok=True)

    print("=" * 70)
    print("  SortGPT Grid Training")
    print(f"  Output: {OUTPUT_BASE}")
    print(f"  Configs: {len(CONFIGS)}")
    print(f"  init_std={INIT_STD}, lr={LR}, seed={SEED}, n_embd={N_EMBD}")
    print(f"  max_iters={MAX_ITERS}, checkpoint_every={CHECKPOINT_EVERY}")
    print("=" * 70)

    # Phase 1: Launch all training jobs
    train_jobs = {}
    for cfg in CONFIGS:
        name = config_dir_name(cfg)
        proc, log_file, log_path = launch_training(cfg)
        train_jobs[name] = {
            "cfg": cfg, "proc": proc, "log_file": log_file, "log_path": log_path,
            "phase": "training",
        }
        print(f"  [GPU {cfg['gpu']}] Started training {name} (PID {proc.pid})")

    print(f"\n  All {len(train_jobs)} training jobs launched.")
    print("  Monitoring...\n")

    # Phase 2: Monitor training, launch plotting when training finishes
    plot_jobs = {}
    t0 = time.time()

    while train_jobs or plot_jobs:
        finished_train = []
        for name, job in train_jobs.items():
            ret = job["proc"].poll()
            if ret is not None:
                job["log_file"].close()
                elapsed = time.time() - t0
                if ret == 0:
                    print(f"  [{elapsed:7.0f}s] [GPU {job['cfg']['gpu']}] "
                          f"Training DONE: {name}")
                    finished_train.append(name)
                else:
                    print(f"  [{elapsed:7.0f}s] [GPU {job['cfg']['gpu']}] "
                          f"Training FAILED: {name} (exit={ret})")
                    print(f"           Log: {job['log_path']}")
                    finished_train.append(name)

        for name in finished_train:
            job = train_jobs.pop(name)
            if job["proc"].returncode == 0:
                cfg = job["cfg"]
                proc, log_file, log_path = launch_plotting(cfg)
                plot_jobs[name] = {
                    "cfg": cfg, "proc": proc, "log_file": log_file, "log_path": log_path,
                }
                elapsed = time.time() - t0
                print(f"  [{elapsed:7.0f}s] [GPU {cfg['gpu']}] "
                      f"Started plotting {name} (PID {proc.pid})")

        finished_plot = []
        for name, job in plot_jobs.items():
            ret = job["proc"].poll()
            if ret is not None:
                job["log_file"].close()
                elapsed = time.time() - t0
                if ret == 0:
                    print(f"  [{elapsed:7.0f}s] [GPU {job['cfg']['gpu']}] "
                          f"Plotting DONE: {name}")
                else:
                    print(f"  [{elapsed:7.0f}s] [GPU {job['cfg']['gpu']}] "
                          f"Plotting FAILED: {name} (exit={ret})")
                    print(f"           Log: {job['log_path']}")
                finished_plot.append(name)

        for name in finished_plot:
            plot_jobs.pop(name)

        if train_jobs or plot_jobs:
            time.sleep(10)

    total_elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"  All done in {total_elapsed / 60:.1f} minutes")
    print(f"  Results in: {OUTPUT_BASE}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
