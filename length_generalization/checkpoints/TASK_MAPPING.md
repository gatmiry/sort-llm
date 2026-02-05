# Task ID to Configuration Mapping

This document explains how task directories map to specific experimental configurations.

## Overview

Each experiment group is organized into **task directories** (`task_0000/`, `task_0001/`, etc.). The task ID is computed by enumerating all combinations of experimental variables in a fixed order.

## Grid Construction

The grid is built using a cross-product of variables in this order:

### Dynamic Training (multi-length)
1. **Layers**: [2, 1] (sorted descending)
2. **MLP**: [True, False]
3. **Length Mode**: ["mix", "curriculum"]

### Fixed Training (single-K)
1. **Layers**: [2, 1] (sorted descending)
2. **MLP**: [True, False]

## Dynamic Training Task Mapping

For experiments like `len_gen_k8` (train K=2→8), `len_gen_k10` (train K=2→10), etc.

Each task directory contains runs for **BOTH MLP configurations** (on and off), with **3 seeds each**.

| Task ID | Layers | Length Mode | Contains |
|---------|--------|-------------|----------|
| **task_0000** | 2 | mix | L2, mix: MLP on (3 seeds) + MLP off (3 seeds) |
| **task_0001** | 2 | curriculum | L2, curriculum: MLP on (3 seeds) + MLP off (3 seeds) |
| **task_0002** | 1 | mix | **L1, mix: MLP on (3 seeds) + MLP off (3 seeds)** ⭐ |
| **task_0003** | 1 | curriculum | L1, curriculum: MLP on (3 seeds) + MLP off (3 seeds) |

⭐ **task_0002** (1-layer, mix) contains the **best generalizing configuration** (no MLP variant).

### Example: What's in task_0002 for len_gen_k8?
```
task_0002/
├── saved_models/
│   ├── Final_..._L1_..._mlp1_lenmix_...pt  # 1-layer WITH MLP, seeds 1337,2337,3337
│   ├── Final_..._L1_..._mlp0_lenmix_...pt  # 1-layer NO MLP, seeds 1337,2337,3337 ⭐
│   └── Checkpoint_...pt (intermediate checkpoints)
└── wandb/  # 6+ W&B runs (MLP on/off × 3 seeds)
```

### Training Details
- All tasks train on K=2→8 (dynamic range) for `len_gen_k8`
- Test on K=17→32 (out-of-distribution)
- Total per task: **6+ W&B runs** (2 MLP configs × 3 seeds, plus any reruns)

## Fixed Training Task Mapping

For experiments like `len_gen_single_k8` (train K=8 only), `len_gen_single_k4` (train K=4 only), etc.

| Task ID | Layers | MLP | Contains |
|---------|--------|-----|----------|
| **task_0000** | 2 | On | L2, MLP on (seeds: 1337, 2337, 3337) |
| **task_0001** | 2 | Off | L2, MLP off (seeds: 1337, 2337, 3337) |
| **task_0002** | 1 | On | L1, MLP on (seeds: 1337, 2337, 3337) |
| **task_0003** | 1 | Off | **L1, MLP off (seeds: 1337, 2337, 3337)** ⭐ |

⭐ **task_0003** achieves remarkable ~98% generalization when trained on just K=4!

### Example: What's in task_0003 for len_gen_single_k4?
```
task_0003/
├── saved_models/
│   ├── Final_..._L1_..._mlp0_..._trainK4to4_...pt  # 3 files (one per seed)
│   └── Checkpoint_...pt (intermediate checkpoints, 3 files)
└── wandb/  # 3-4 W&B runs (one per seed, plus any reruns)
```

### Training Details
- All tasks train on fixed K=4 only for `len_gen_single_k4`
- Test on K=17→32 (out-of-distribution)
- Total per task: **3 W&B runs** (one per seed)

## Without LayerNorm Variants

Same task structure applies to no-layernorm experiments:
- `len_gen_no_ln_k{N}/` → dynamic training, no layernorm
- `len_gen_single_no_ln_k{N}/` → fixed single-K, no layernorm

## Locating Checkpoints for a Specific Configuration

### Example 1: Find "1-layer, no MLP, mix, train K=2→8"
```
checkpoints/with_layernorm/dynamic/len_gen_k8/task_0002/
  → Look for files with mlp0 in the filename
```

### Example 2: Find "1-layer, no MLP, train K=4 only"
```
checkpoints/with_layernorm/fixed/len_gen_single_k4/task_0003/
  → All files in this directory are L1, MLP off
```

### Example 3: Find "2-layer, with MLP, curriculum, train K=2→12"
```
checkpoints/with_layernorm/dynamic/len_gen_k12/task_0001/
  → Look for files with mlp1 in the filename
```

### Example 4: Find "1-layer, with MLP, mix, train K=2→8"
```
checkpoints/with_layernorm/dynamic/len_gen_k8/task_0002/
  → Look for files with mlp1 in the filename
```

## Checkpoint Files Within Each Task

Each task directory contains:
```
task_XXXX/
├── saved_models/
│   ├── Checkpoint_*.pt      # Intermediate checkpoint @ iter 20k
│   └── Final_*.pt            # Final checkpoint @ iter 40k
└── wandb/                    # W&B logs and metrics
    └── wandb/
        └── run-YYYYMMDD_HHMMSS-{run_id}/
            ├── files/
            │   ├── wandb-summary.json
            │   ├── config.yaml
            │   └── wandb-history.jsonl
            └── logs/
```

## Seeds and Multiple Runs

Each task contains checkpoints from **3 independent runs** with different random seeds:
- Seed 1337
- Seed 2337
- Seed 3337

The seed is embedded in the W&B run but not in the checkpoint filename. To identify which seed a checkpoint corresponds to, check the corresponding `config.yaml` file in the W&B directory.

## Quick Reference: Best Generalizing Configurations

Based on our results, these configurations show the strongest out-of-distribution generalization:

| Configuration | Location | Gen Performance (K17-32) |
|--------------|----------|------------------------|
| 1L, no MLP, fixed K=4 | `fixed/len_gen_single_k4/task_0003/` | ~98% |
| 1L, no MLP, dynamic mix K=2→6 | `dynamic/len_gen_k6/task_0006/` | ~96% |
| 1L, no MLP, dynamic mix K=2→8 | `dynamic/len_gen_k8/task_0006/` | ~100% |
| 2L, with MLP, dynamic mix K=2→8 | `dynamic/len_gen_k8/task_0000/` | ~96% |

## Programmatic Task Decoding

If you need to decode task IDs programmatically:

```python
def decode_dynamic_task(task_id):
    """Decode task ID for dynamic training."""
    layer_counts = [2, 1]
    use_mlp_flags = [True, False]
    length_modes = ["mix", "curriculum"]
    
    combos = []
    for L in layer_counts:
        for mlp_on in use_mlp_flags:
            for lm in length_modes:
                combos.append((L, mlp_on, lm))
    
    if 0 <= task_id < len(combos):
        L, mlp, mode = combos[task_id]
        return {"layers": L, "mlp": mlp, "length_mode": mode}
    return None

def decode_fixed_task(task_id):
    """Decode task ID for fixed single-K training."""
    layer_counts = [2, 1]
    use_mlp_flags = [True, False]
    
    combos = []
    for L in layer_counts:
        for mlp_on in use_mlp_flags:
            combos.append((L, mlp_on))
    
    if 0 <= task_id < len(combos):
        L, mlp = combos[task_id]
        return {"layers": L, "mlp": mlp}
    return None

# Example usage
print(decode_dynamic_task(6))  # {"layers": 1, "mlp": False, "length_mode": "mix"}
print(decode_fixed_task(3))    # {"layers": 1, "mlp": False}
```
