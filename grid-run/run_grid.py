"""
Grid search launcher. Runs multiple jobs per GPU to maximize utilization.
Estimates memory per job based on vocab_size and assigns accordingly.

Grid:
  vocab_size:     [64, 128, 256, 512, 8192]
  block_size:     [16, 32]
  max_iters:      [10000, 20000, 40000, 60000]
  learning_rate:  [1e-4, 1e-3, 1e-2]
  with_layer_norm:[0, 1]

Fixed: n_layers=2, n_heads=1, n_embd=64, batch_size=4096, without_pos=True
Total: 5 * 2 * 4 * 3 * 2 = 240 runs
"""
import itertools
import os
import subprocess
import sys
import time

NUM_GPUS = 8
GPU_MEM_MB = 78000  # usable memory per GPU in MB (leave headroom from 81559)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_BASE = os.path.join(SCRIPT_DIR, 'outputs')

VOCAB_SIZES = [64, 128, 256, 512, 8192]
BLOCK_SIZES = [16, 32]
MAX_ITERS = [10000, 20000, 40000, 60000]
LEARNING_RATES = [1e-4, 1e-3, 1e-2]
LAYER_NORMS = [0, 1]

# Estimated GPU memory per job (MB) based on vocab_size
MEM_ESTIMATE = {64: 3000, 128: 3500, 256: 4000, 512: 5000, 8192: 25000}


def make_configs():
    configs = []
    for vs, bs, mi, lr, ln in itertools.product(
        VOCAB_SIZES, BLOCK_SIZES, MAX_ITERS, LEARNING_RATES, LAYER_NORMS
    ):
        lr_str = f"{lr:.0e}"
        run_name = f"V{vs}_B{bs}_LR{lr_str}_MI{mi}_LN{ln}_E64_H1_L2"
        out_dir = os.path.join(OUTPUT_BASE, run_name)
        configs.append({
            'vocab_size': vs, 'block_size': bs, 'max_iters': mi,
            'learning_rate': lr, 'with_layer_norm': ln,
            'run_name': run_name, 'output_dir': out_dir,
            'est_mem': MEM_ESTIMATE.get(vs, 5000),
        })
    return configs


def is_completed(cfg):
    if not os.path.exists(cfg['output_dir']):
        return False
    existing = os.listdir(cfg['output_dir'])
    final_ckpt = f"_itr{cfg['max_iters']}.pt"
    return any(final_ckpt in f for f in existing)


def launch_job(cfg, gpu_id):
    cmd = [
        sys.executable, os.path.join(SCRIPT_DIR, 'train_single.py'),
        '--vocab_size', str(cfg['vocab_size']),
        '--block_size', str(cfg['block_size']),
        '--max_iters', str(cfg['max_iters']),
        '--learning_rate', str(cfg['learning_rate']),
        '--with_layer_norm', str(cfg['with_layer_norm']),
        '--gpu', '0',
        '--output_dir', cfg['output_dir'],
    ]
    log_dir = os.path.join(OUTPUT_BASE, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{cfg['run_name']}.log")
    log_file = open(log_path, 'w')
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    proc = subprocess.Popen(
        cmd, stdout=log_file, stderr=subprocess.STDOUT,
        env=env, cwd=SCRIPT_DIR,
    )
    return proc, log_file


def main():
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    configs = make_configs()

    # Sort: shortest jobs first (small max_iters, small vocab) for faster turnover
    configs.sort(key=lambda c: (c['max_iters'], c['vocab_size']))

    total = len(configs)
    print(f"Total configurations: {total}")
    print(f"Using {NUM_GPUS} GPUs, {GPU_MEM_MB}MB usable each")

    # Track per-GPU: list of (proc, log_file, run_name, est_mem)
    gpu_jobs = {g: [] for g in range(NUM_GPUS)}
    gpu_mem_used = {g: 0 for g in range(NUM_GPUS)}

    queue = list(configs)
    completed = 0
    failed = 0
    skipped = 0
    t0 = time.time()

    while queue or any(gpu_jobs[g] for g in range(NUM_GPUS)):
        # Check for finished jobs on each GPU
        for g in range(NUM_GPUS):
            still_running = []
            for proc, log_file, name, mem in gpu_jobs[g]:
                ret = proc.poll()
                if ret is not None:
                    log_file.close()
                    gpu_mem_used[g] -= mem
                    if ret == 0:
                        completed += 1
                        elapsed = time.time() - t0
                        print(f"[DONE] GPU {g}: {name} (done={completed} fail={failed} queue={len(queue)} | {elapsed:.0f}s)")
                    else:
                        failed += 1
                        print(f"[FAIL] GPU {g}: {name} exit={ret} (done={completed} fail={failed} queue={len(queue)})")
                else:
                    still_running.append((proc, log_file, name, mem))
            gpu_jobs[g] = still_running

        # Try to assign queued jobs to GPUs with available memory
        remaining_queue = []
        for cfg in queue:
            if is_completed(cfg):
                skipped += 1
                continue
            placed = False
            # Find GPU with most free memory that can fit this job
            best_gpu = None
            best_free = -1
            for g in range(NUM_GPUS):
                free = GPU_MEM_MB - gpu_mem_used[g]
                if free >= cfg['est_mem'] and free > best_free:
                    best_gpu = g
                    best_free = free
            if best_gpu is not None:
                proc, log_file = launch_job(cfg, best_gpu)
                gpu_jobs[best_gpu].append((proc, log_file, cfg['run_name'], cfg['est_mem']))
                gpu_mem_used[best_gpu] += cfg['est_mem']
                n_on_gpu = len(gpu_jobs[best_gpu])
                print(f"[LAUNCH] GPU {best_gpu} ({n_on_gpu} jobs, {gpu_mem_used[best_gpu]}MB): {cfg['run_name']}")
                placed = True
            if not placed:
                remaining_queue.append(cfg)
        queue = remaining_queue

        if any(gpu_jobs[g] for g in range(NUM_GPUS)):
            time.sleep(3)

    elapsed = time.time() - t0
    print(f"\nAll done! completed={completed}, failed={failed}, skipped={skipped}, elapsed={elapsed:.0f}s")


if __name__ == '__main__':
    main()
