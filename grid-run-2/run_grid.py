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
import argparse
import itertools
import os
import subprocess
import sys
import time

NUM_GPUS = 8
GPU_MEM_TOTAL = 81559
MEM_CAP_FRAC = 0.80  # only use 80% of each GPU to avoid contention
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_BASE = os.path.join(SCRIPT_DIR, 'outputs')

VOCAB_SIZES = [64, 128, 256, 512, 8192]
BLOCK_SIZES = [16, 32]
MAX_ITERS = [10000, 20000, 40000, 60000]
LEARNING_RATES = [1e-4, 1e-3, 1e-2]
LAYER_NORMS = [0, 1]

# Estimated GPU memory per job (MB) based on actual usage + headroom for PyTorch pool
MEM_ESTIMATE = {64: 4000, 128: 4500, 256: 5000, 512: 6000, 8192: 28000}


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


def get_gpu_mem_used():
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'],
            text=True)
        usage = {}
        for line in out.strip().split('\n'):
            parts = line.split(',')
            usage[int(parts[0].strip())] = int(parts[1].strip())
        return usage
    except Exception:
        return {i: 0 for i in range(NUM_GPUS)}


def launch_job(cfg, gpu_id, seed=None):
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
    if seed is not None:
        cmd += ['--seed', str(seed)]
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility (passed to train_single.py)')
    args = parser.parse_args()

    os.makedirs(OUTPUT_BASE, exist_ok=True)
    configs = make_configs()

    configs.sort(key=lambda c: (c['max_iters'], c['vocab_size']))

    total = len(configs)
    mem_cap = int(GPU_MEM_TOTAL * MEM_CAP_FRAC)
    print(f"Total configurations: {total}")
    print(f"Using {NUM_GPUS} GPUs, cap {mem_cap}MB each ({MEM_CAP_FRAC*100:.0f}% of {GPU_MEM_TOTAL}MB)")
    if args.seed is not None:
        print(f"Seed: {args.seed}")

    gpu_jobs = {g: [] for g in range(NUM_GPUS)}
    gpu_own_mem = {g: 0 for g in range(NUM_GPUS)}

    queue = list(configs)
    completed = 0
    failed = 0
    skipped = 0
    t0 = time.time()

    while queue or any(gpu_jobs[g] for g in range(NUM_GPUS)):
        for g in range(NUM_GPUS):
            still_running = []
            for proc, log_file, name, mem in gpu_jobs[g]:
                ret = proc.poll()
                if ret is not None:
                    log_file.close()
                    gpu_own_mem[g] -= mem
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

        actual_mem = get_gpu_mem_used()

        remaining_queue = []
        for cfg in queue:
            if is_completed(cfg):
                skipped += 1
                continue
            placed = False
            best_gpu = None
            best_headroom = -1
            for g in range(NUM_GPUS):
                external = max(0, actual_mem.get(g, 0) - gpu_own_mem[g])
                used = gpu_own_mem[g] + external
                headroom = mem_cap - used
                if headroom >= cfg['est_mem'] and headroom > best_headroom:
                    best_gpu = g
                    best_headroom = headroom
            if best_gpu is not None:
                proc, log_file = launch_job(cfg, best_gpu, seed=args.seed)
                gpu_jobs[best_gpu].append((proc, log_file, cfg['run_name'], cfg['est_mem']))
                gpu_own_mem[best_gpu] += cfg['est_mem']
                n_on_gpu = len(gpu_jobs[best_gpu])
                print(f"[LAUNCH] GPU {best_gpu} ({n_on_gpu} jobs, est_alloc={gpu_own_mem[best_gpu]}MB): {cfg['run_name']}")
                placed = True
            if not placed:
                remaining_queue.append(cfg)
        queue = remaining_queue

        if any(gpu_jobs[g] for g in range(NUM_GPUS)):
            time.sleep(5)

    elapsed = time.time() - t0
    print(f"\nAll done! completed={completed}, failed={failed}, skipped={skipped}, elapsed={elapsed:.0f}s")


if __name__ == '__main__':
    main()
