# Checkpoint Organization

This directory contains model checkpoints from length generalization experiments.

## Directory Structure

```
checkpoints/
├── with_layernorm/          # Standard runs with LayerNorm
│   ├── dynamic/            # Dynamic multi-length training
│   │   ├── len_gen_k4/    # train_max_k=4 (K=2->4)
│   │   │   ├── task_0000/ # L2, mix (contains MLP on + off × 3 seeds each)
│   │   │   ├── task_0001/ # L2, curriculum (contains MLP on + off × 3 seeds each)
│   │   │   ├── task_0002/ # L1, mix (contains MLP on + off × 3 seeds each) ⭐
│   │   │   └── task_0003/ # L1, curriculum (contains MLP on + off × 3 seeds each)
│   │   ├── len_gen_k6/    # train_max_k=6 (K=2->6), same 4-task structure
│   │   ├── len_gen_k8/
│   │   ├── len_gen_k10/
│   │   ├── len_gen_k12/
│   │   ├── len_gen_k14/
│   │   └── len_gen_k16/
│   └── fixed/              # Fixed single-K training
│       ├── len_gen_single_k4/  # train at K=4 only
│       │   ├── task_0000/ # L2, MLP on (3 seeds)
│       │   ├── task_0001/ # L2, MLP off (3 seeds)
│       │   ├── task_0002/ # L1, MLP on (3 seeds)
│       │   └── task_0003/ # L1, MLP off (3 seeds) ⭐
│       ├── len_gen_single_k6/  # same 4-task structure
│       └── ...
├── without_layernorm/       # No LayerNorm experiments (in progress)
│   ├── dynamic/
│   │   ├── len_gen_no_ln_k4/
│   │   └── ...
│   └── fixed/
│       ├── len_gen_single_no_ln_k4/
│       └── ...
├── TASK_MAPPING.md          # Detailed task ID → configuration mapping
└── README.md                # This file
```

⭐ Best generalizing configurations typically in task_0002 (L1, mix, MLP off files) or task_0003 (L1, MLP off fixed)

## Task Directory Structure

Each experiment group (e.g., `len_gen_k8/`) contains multiple **task directories** (`task_0000/`, `task_0001/`, etc.). Each task represents a unique combination of:
- Number of layers (1 or 2)
- MLP enabled or disabled
- Data loading mode (mix or curriculum) [for dynamic training only]

The task IDs are assigned systematically based on a grid cross-product.

### Dynamic Training Task Mapping

For dynamic training (e.g., `len_gen_k8`), there are **8 tasks per seed** (4 base configs × 2 length modes):

| Task ID | Layers | MLP | Length Mode |
|---------|--------|-----|-------------|
| task_0000 | 2 | On | mix |
| task_0001 | 2 | On | curriculum |
| task_0002 | 2 | Off | mix |
| task_0003 | 2 | Off | curriculum |
| task_0004 | 1 | On | mix |
| task_0005 | 1 | On | curriculum |
| task_0006 | 1 | Off | mix |
| task_0007 | 1 | Off | curriculum |

Each task directory contains **3 independent runs** (seeds: 1337, 2337, 3337), each with multiple checkpoints.

### Fixed Training Task Mapping

For fixed single-K training (e.g., `len_gen_single_k8`), there are **4 tasks per seed** (no length mode variation):

| Task ID | Layers | MLP |
|---------|--------|-----|
| task_0000 | 2 | On |
| task_0001 | 2 | Off |
| task_0002 | 1 | On |
| task_0003 | 1 | Off |

Each task directory contains **3 independent runs** (seeds: 1337, 2337, 3337).

## Checkpoint Naming Convention

Checkpoints follow this naming pattern:

```
{Prefix}_N{vocab}_d{embd}_H{heads}_L{layers}_npos{0|1}_mlp{0|1}_ln{0|1}_len{mode}_trainK{min}to{max}_testK{test_min}to{test_max}_nodup{0|1}_iters{step}.pt
```

### Components

| Field | Description | Values |
|-------|-------------|--------|
| **Prefix** | Checkpoint type | `Checkpoint` (intermediate, iter 20k) or `Final` (end of training, iter 40k) |
| **N** | Vocabulary size | 128 (fixed) |
| **d** | Embedding dimension | 128 (fixed) |
| **H** | Number of attention heads | 1 (fixed) |
| **L** | Number of transformer layers | 1 or 2 |
| **npos** | Positional embedding disabled | 1 (always disabled in these experiments) |
| **mlp** | MLP enabled | 0 = attention-only, 1 = full transformer block |
| **ln** | LayerNorm enabled | 0 = no normalization, 1 = standard LayerNorm |
| **len** | Length sampling mode | `mix` (uniform) or `curriculum` (progressive) |
| **trainK** | Training length range | e.g., `K2to8` (dynamic) or `K4to4` (fixed single-K) |
| **testK** | Test length range | `K17to32` (always out-of-distribution) |
| **nodup** | No duplicates in data | 1 (always enforced) |
| **iters** | Training iteration | 20001 (checkpoint) or 40000 (final) |

### Examples

#### Example 1: Simple Attention-Only Model
```
Final_N128_d128_H1_L1_npos1_mlp0_ln1_lenmix_trainK4to4_testK17to32_nodup1_iters40000.pt
```
- 1 layer, attention-only (no MLP), with LayerNorm
- Uniform mixing data loading
- Trained on **fixed K=4 only**
- Final checkpoint at 40k iterations
- **This configuration achieves 98% generalization to K=17-32!**

#### Example 2: Full Transformer with Curriculum
```
Final_N128_d128_H1_L2_npos1_mlp1_ln1_lencurriculum_trainK2to12_testK17to32_nodup1_iters40000.pt
```
- 2 layers, with MLP, with LayerNorm
- Curriculum data loading (progressive K=2→12)
- Trained on **dynamic K=2-12**
- Final checkpoint at 40k iterations

#### Example 3: No LayerNorm
```
Final_N128_d128_H1_L1_npos1_mlp0_ln0_lenmix_trainK2to8_testK17to32_nodup1_iters40000.pt
```
- 1 layer, attention-only, **no LayerNorm**
- Uniform mixing, dynamic K=2-8

## Checkpoint Contents

Each `.pt` file contains:
```python
{
    'model_state_dict': OrderedDict,  # Model weights
    'optimizer_state_dict': dict,     # Optimizer state
    'config': TrainConfig,            # Full training configuration
    'iters_done': int,               # Number of iterations completed
}
```

## Loading Checkpoints

```python
import torch
from sortGPT_len_generalization import GPT, GPTConfig

# Load checkpoint
ckpt = torch.load('path/to/checkpoint.pt', map_location='cpu')

# Reconstruct model config
cfg = ckpt['config']
model_cfg = GPTConfig(
    block_size=cfg.train_max_k,
    vocab_size=cfg.vocab_n + 1,  # +1 for separator
    n_layers=cfg.n_layers,
    n_heads=cfg.n_heads,
    n_embd=cfg.n_embd,
    without_pos=cfg.without_pos,
    use_mlp=cfg.use_mlp,
    use_layernorm=cfg.use_layernorm,
    max_seq_len=2 * cfg.test_max_k + 1,
)

# Create and load model
model = GPT(model_cfg)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
```

## Checkpoint Statistics

### Storage per checkpoint
- Intermediate (iter 20k): ~1-2 MB per checkpoint
- Final (iter 40k): ~1-2 MB per checkpoint
- Total checkpoints: ~496 files (2 per run × 248 configurations)
- Estimated total size: ~500 MB - 1 GB

### Organization by experiment
Each experimental condition has **3 seeds × 2 checkpoints** = 6 files:
- 3 independent runs (seeds: 1337, 2337, 3337)
- 2 checkpoints per run (intermediate @ 20k, final @ 40k)

## Adding New Checkpoints

When adding checkpoints from new runs:

1. **Identify the experiment group** (with/without layernorm, dynamic/fixed)
2. **Create appropriate subdirectory** if needed
3. **Use consistent naming** following the convention above
4. **Update this README** if new experiments are added

## Source Location

Original checkpoints are stored at:
```
/temp_work/ch225816/sort-llm/grid_outputs/
```

Organized by W&B group name:
- `len_gen_k{N}/` - dynamic with layernorm
- `len_gen_single_k{N}/` - fixed with layernorm  
- `len_gen_no_ln_k{N}/` - dynamic without layernorm
- `len_gen_single_no_ln_k{N}/` - fixed without layernorm

Each group contains task directories with saved checkpoints and W&B logs.
