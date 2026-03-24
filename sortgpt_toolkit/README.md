# SortGPT Toolkit

Tools for training and analyzing SortGPT models — small transformers that learn to sort sequences of integers.

## Setup

Requires Python 3.8+ with:
```
pip install torch numpy pandas matplotlib tqdm
```

## Quick Start

### 1. Train a single model

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
    --init-seed 1501 --init-std 0.02 --run-dir ./runs/test \
    --max-iters 100000 --checkpoint-every 20000
```

Key flags:
- `--init-seed`: Seed for weight init + RNG (different seeds → different runs)
- `--init-std`: Std dev for weight initialization (e.g. 0.02, 0.03)
- `--max-iters`: Total training steps
- `--checkpoint-every`: Save model every N steps
- `--lr`: Learning rate (default: 0.03)

### 2. Run a multi-seed experiment

Edit the **CONFIGURATION** section in `run_experiment.sh`:

```bash
SEEDS=(1501 1502 1503 1504 1505)
INIT_STDS=(0.02 0.02 0.02 0.02 0.02)
MAX_ITERS=1000000
CHECKPOINT_EVERY=50000
TRAIN_GPUS=(0 1 2 3 4)
MONITOR_GPU=5
EXPERIMENT_NAME="5seeds_std002"
```

Then run:
```bash
bash run_experiment.sh
```

This launches one training process per GPU plus a live monitor that:
- Evaluates ablation (full / no-attn-1 / no-attn-2) at each checkpoint
- Evaluates length generalization across 26 lengths (4–256)
- Auto-kills seeds that diverge
- Writes `status.txt` with GREEN/RED color coding
- Generates plots only once ALL alive seeds reach each checkpoint

### 3. Managing a running experiment

```bash
# Check status (color-coded by no_attn2 result)
cat runs/my_experiment_*/status.txt

# Watch monitor output
tail -f runs/my_experiment_*/logs/monitor.log

# Stop one seed
kill $(cat runs/my_experiment_*/pids/seed_1503.pid)

# Stop everything
kill $(cat runs/my_experiment_*/pids/*.pid)
```

### 4. Post-hoc evaluation and plotting

For runs without a live monitor, or to regenerate plots:

```bash
CUDA_VISIBLE_DEVICES=0 python plot_results.py \
    --run-dir ./runs/my_experiment_* \
    --seeds 1501 1502 1503 1504 1505 \
    --init-stds 0.02 0.02 0.02 0.02 0.02 \
    --checkpoint-every 20000
```

Produces:
- `plots/lengthgen_all_checkpoints.png` — all seeds at each checkpoint
- `plots/lengthgen_per_seed_progression.png` — per-seed training progression
- `plots/ablation_ckpt{N}.png` — ablation bar charts at each checkpoint

### 5. Attention heatmaps

Two types of heatmaps showing how models attend:

```bash
# From a run directory (auto-discovers checkpoints)
CUDA_VISIBLE_DEVICES=0 python plot_heatmaps.py \
    --run-dir ./runs/my_experiment_* \
    --output-dir ./runs/my_experiment_*/heatmaps

# Or specify checkpoints manually
CUDA_VISIBLE_DEVICES=0 python plot_heatmaps.py \
    --checkpoints path/to/ckpt1.pt path/to/ckpt2.pt \
    --labels "seed=1501" "seed=1502" \
    --output-dir ./heatmaps
```

Produces:
- **Positional heatmaps** (33×33): Full attention pattern for one sample, axes reordered by token value
- **Averaged heatmaps** (256×256): Sorted→unsorted attention averaged over 1000 samples, indexed by token value

## File Structure

```
sortgpt_toolkit/
├── README.md              ← You are here
├── model.py               ← Model definition, data generation, shared utilities
├── evaluate.py            ← Ablation analysis and length generalization
├── train.py               ← Single-model training script
├── run_experiment.sh      ← Multi-seed launcher (edit config, then run)
├── monitor.py             ← Live monitoring during training
├── plot_results.py        ← Post-hoc evaluation and plotting
└── plot_heatmaps.py       ← Attention pattern visualization
```

## Output Directory Structure

Each experiment produces:
```
runs/my_experiment_YYYYMMDD_HHMMSS/
├── checkpoints/           ← Model checkpoints (.pt files)
│   ├── std0p02_iseed1501__ckpt50000.pt
│   ├── std0p02_iseed1501__ckpt100000.pt
│   └── ...
├── plots/                 ← Generated plots
│   ├── ablation_ckpt50000.png
│   ├── lengthgen_ckpt50000.png
│   └── ...
├── logs/                  ← Training and monitor logs
│   ├── seed_1501_gpu0.log
│   └── monitor.log
├── pids/                  ← PID files for process management
│   ├── seed_1501.pid
│   └── monitor.pid
└── status.txt             ← Color-coded status summary
```

## Color Coding

Throughout the toolkit, models are classified by their attention ablation results:

- **GREEN**: `no_attn2 > 0.95` — The model solves sorting using only layer 1 attention (single-layer solution). Layer 2 attention can be removed with no accuracy loss.
- **RED**: `no_attn2 < 0.95` — The model requires both attention layers (two-layer solution).
- **DIVERGED**: Model failed to learn (0% accuracy). Automatically killed by the monitor.

## Architecture

Default model: 2-layer transformer, 1 attention head, 64-dim embeddings, no positional embeddings, with MLP and final LayerNorm. Trained on the sorting task: given 16 random integers from [0, 255], predict them in sorted order.

All architecture parameters are configurable via `train.py` flags:
```
--block-size 16    --vocab-n 256    --n-layers 2
--n-heads 1        --n-embd 64      --use-pos
--no-mlp           --no-final-ln
```
