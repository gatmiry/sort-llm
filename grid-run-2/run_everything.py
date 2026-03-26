"""
Master script: trains all configs with seed=42, then runs full analysis pipeline.
Phases:
  1. Training grid (run_grid.py --seed 42)
  2. Base analysis: cinclogits + intensity (run_all_analysis.py)
  3. Intensity with ub=10 and ub=15
  4. Ablation (skip attention per layer)
  5. Baseline accuracy (intact model)
"""
import os
import subprocess
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SEED = 42


def run_phase(description, cmd):
    print(f"\n{'='*70}")
    print(f"PHASE: {description}")
    print(f"CMD:   {' '.join(cmd)}")
    print(f"{'='*70}", flush=True)
    t0 = time.time()
    ret = subprocess.call(cmd, cwd=SCRIPT_DIR)
    elapsed = time.time() - t0
    status = "OK" if ret == 0 else f"FAILED (exit={ret})"
    print(f"\n>>> {description}: {status} ({elapsed:.0f}s)\n", flush=True)
    return ret


def main():
    py = sys.executable
    t_start = time.time()

    run_phase("Training grid (seed=42)", [
        py, os.path.join(SCRIPT_DIR, 'run_grid.py'), '--seed', str(SEED),
    ])

    run_phase("Base analysis (cinclogits + intensity)", [
        py, os.path.join(SCRIPT_DIR, 'run_all_analysis.py'),
    ])

    run_phase("Intensity ub=10,15", [
        py, os.path.join(SCRIPT_DIR, 'run_all_analysis.py'),
        '--intensity-ub', '10', '15',
    ])

    run_phase("Ablation analysis", [
        py, os.path.join(SCRIPT_DIR, 'run_all_analysis.py'), '--ablation',
    ])

    run_phase("Baseline accuracy", [
        py, os.path.join(SCRIPT_DIR, 'run_all_analysis.py'), '--baseline',
    ])

    total = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"ALL DONE — total elapsed: {total:.0f}s ({total/3600:.1f}h)")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
