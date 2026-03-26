"""
Parallel launcher for aggressive-intervention experiments across all
V512_B32 configs (varying LR and MI). Distributes across 8 GPUs.
"""
import os
import subprocess
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKER_SCRIPT = os.path.join(SCRIPT_DIR, 'run_config_all.py')
NUM_GPUS = 8

LEARNING_RATES = ['1e-04', '1e-03', '1e-02']
MAX_ITERS = [10000, 20000, 40000, 60000]

ALREADY_DONE = {'V512_B32_LR1e-02_MI20000'}


def is_done(cfg_name):
    out_dir = os.path.join(SCRIPT_DIR, f"{cfg_name}_E64_H1_L2")
    expected = [
        'aggressive_intensity_layer0.png', 'aggressive_intensity_layer1.png',
        'aggressive_sep_intensity_layer0.png', 'aggressive_sep_intensity_layer1.png',
    ]
    return all(os.path.exists(os.path.join(out_dir, f)) for f in expected)


def main():
    configs = []
    for lr in LEARNING_RATES:
        for mi in MAX_ITERS:
            cfg = f"V512_B32_LR{lr}_MI{mi}"
            if cfg in ALREADY_DONE:
                print(f"  SKIP (already done): {cfg}")
                continue
            if is_done(cfg):
                print(f"  SKIP (plots exist): {cfg}")
                continue
            configs.append(cfg)

    print(f"Configs to run: {len(configs)}")
    for c in configs:
        print(f"  {c}")

    gpu_procs = {g: None for g in range(NUM_GPUS)}
    queue = list(configs)
    completed = 0
    failed = 0
    t0 = time.time()

    while queue or any(gpu_procs[g] is not None for g in range(NUM_GPUS)):
        for g in range(NUM_GPUS):
            if gpu_procs[g] is not None:
                proc, log_file, name = gpu_procs[g]
                ret = proc.poll()
                if ret is not None:
                    log_file.close()
                    elapsed = time.time() - t0
                    if ret == 0:
                        completed += 1
                        print(f"[DONE] GPU {g}: {name} (done={completed} fail={failed} queue={len(queue)} | {elapsed:.0f}s)")
                    else:
                        failed += 1
                        print(f"[FAIL] GPU {g}: {name} exit={ret}")
                    gpu_procs[g] = None

        for g in range(NUM_GPUS):
            if gpu_procs[g] is None and queue:
                cfg = queue.pop(0)
                cmd = [sys.executable, WORKER_SCRIPT, cfg, '--device', 'cuda', '--aggressive-only']
                log_dir = os.path.join(SCRIPT_DIR, 'logs')
                os.makedirs(log_dir, exist_ok=True)
                log_path = os.path.join(log_dir, f"{cfg}.log")
                log_file = open(log_path, 'w')
                env = os.environ.copy()
                env['CUDA_VISIBLE_DEVICES'] = str(g)
                proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT,
                                        env=env, cwd=SCRIPT_DIR)
                gpu_procs[g] = (proc, log_file, cfg)
                print(f"[LAUNCH] GPU {g}: {cfg}")

        if any(gpu_procs[g] is not None for g in range(NUM_GPUS)):
            time.sleep(5)

    elapsed = time.time() - t0
    print(f"\nAll done! completed={completed}, failed={failed}, elapsed={elapsed:.0f}s ({elapsed/60:.1f}m)")


if __name__ == '__main__':
    main()
