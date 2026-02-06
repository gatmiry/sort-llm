# Length Generalization Experiments for Sorting Task

This directory contains code, checkpoints, and results for systematic length generalization experiments on small transformer models trained to sort sequences of numbers.

## Overview

We trained small GPT-style transformers on sorting tasks and systematically evaluated their ability to generalize from short training sequences to much longer test sequences (K=17-32). The experiments explore how different architectural choices, data loading strategies, and training regimes affect length generalization.

## Directory Structure

```
length_generalization/
├── code/                          # Training and analysis scripts
│   ├── sortGPT_len_generalization.py       # Main training script
│   ├── plot_len_generalization_results.py  # Plotting script (basic)
│   ├── plot_comprehensive_results.py       # Comprehensive analysis and plotting
│   ├── plot_layernorm_comparison.py        # LayerNorm comparison plots
│   ├── plot_in_dist_and_layernorm_value.py # In-dist vs OOD scatter & effect size
│   ├── plot_all_final.py                   # Final clean plots + CSVs
│   ├── run_len_generalization.sh          # SLURM script: dynamic training (MLP off)
│   ├── run_len_generalization_single_k.sh # SLURM script: fixed single-K training
│   ├── run_len_generalization_no_ln.sh    # SLURM script: no layer norm
│   └── run_len_generalization_single_k_no_ln.sh  # SLURM script: single-K, no layer norm
├── checkpoints/                   # Model checkpoints (organized by experiment)
│   ├── with_layernorm/           # Standard runs with LayerNorm (~776MB)
│   │   ├── dynamic/              # Multi-length training (len_gen_k4...k16)
│   │   └── fixed/                # Single-K training (len_gen_single_k4...k16)
│   ├── without_layernorm/        # Runs without LayerNorm
│   │   ├── dynamic/              # Multi-length, no layernorm
│   │   └── fixed/                # Single-K, no layernorm
│   ├── README.md                 # Checkpoint organization guide
│   └── TASK_MAPPING.md           # Task ID → configuration mapping
├── results/                       # Plots, CSVs, and analysis outputs
│   ├── fig1_main_result.png
│   ├── fig2_layernorm_dynamic.png
│   ├── fig3_layernorm_fixed.png
│   ├── fig4_in_dist_vs_ood.png
│   ├── fig5_layernorm_effect_size.png
│   ├── summary.csv
│   └── min_k_thresholds.csv
└── README.md                      # This file
```

**Note**: Checkpoint `.pt` files are excluded from git by default (see `.gitignore`). The directory structure and metadata are included, but the actual model weights need to be synced separately or added with `git lfs`.

## Experimental Setup

### Task
- **Input**: Sequence of N random numbers from vocabulary [0, 127]
- **Output**: Sorted sequence (ascending order)
- **Format**: `[num1, num2, ..., numN] SEP [sorted1, sorted2, ..., sortedN]`

### Model Architecture
- **Base**: GPT-style decoder-only transformer
- **Vocabulary**: 128 numbers + 1 separator token = 129 total
- **Embedding dim**: 128
- **Heads**: 1 (single-head attention)
- **Layers**: 1 or 2
- **No positional embeddings** (all experiments)

### Variables Explored

#### 1. **Model Configuration**
- **Layers**: 1 vs 2
- **MLP**: On (full transformer block) vs Off (attention-only)
- **LayerNorm**: On (standard) vs Off (no normalization)

#### 2. **Data Loading Strategy**
- **Mix**: Each batch samples K uniformly from [train_min_k, train_max_k]
- **Curriculum**: K sampled from [train_min_k, max_k(t)] where max_k grows from min→max over training

#### 3. **Training Regime**
- **Dynamic**: Train on range of lengths (e.g., K=2→8)
- **Fixed**: Train on single length only (e.g., K=4)

#### 4. **Training Length Range**
- **train_max_k**: 4, 6, 8, 10, 12, 14, 16
- **train_min_k**: 2 (for dynamic), same as max (for fixed)

### Evaluation
- **Test lengths**: K=17-32 (always out-of-distribution)
- **Metrics**: 
  - Exact match accuracy (entire sequence correct)
  - Token accuracy (per-position correctness)
- **Samples**: 100 per length

### Training Details
- **Iterations**: 40,000
- **Warmup**: 400 iterations
- **Learning rate**: 1e-4 → 1e-6 (cosine decay)
- **Batch size**: 4096 samples
- **Seeds**: 3 independent runs per configuration

## Key Findings

### 1. Simplest Model Generalizes Best from Minimal Data
**Result**: 1-layer attention-only (no MLP) trained on just K=4 achieves **98% exact-match accuracy on K=17-32**.

This **outperforms** dynamic training at the same max K:
- Fixed K=4: 97.9% ± 2.4%
- Dynamic K2→4 (mix): 90.2% ± 3.6%
- Dynamic K2→4 (curriculum): 67.9% ± 1.3%

**Implication**: Pure attention learns a surprisingly general sorting algorithm from minimal examples. Data diversity doesn't always help—focused training on a single (harder) length can be more effective for simple models.

### 2. MLP and Depth Hurt Generalization at Small K
At training max K=4:
- **L1, no MLP**: 97.9% generalization ✓
- **L1, with MLP**: 62.2%
- **L2, no MLP**: 1.9% ✗
- **L2, with MLP**: 0.7% ✗

Deeper/wider models need more diverse training data (K≥10-12) to reach 95%+ generalization.

### 3. Mix > Curriculum at Small K
Dynamic **mix** consistently converges faster than **curriculum** at small training lengths:
- K=6: mix 96.3% vs curriculum 87.2%
- K=8: mix 99.7% vs curriculum 97.5%

Curriculum catches up at larger K (≥12).

### 4. Dynamic vs Fixed Training
For **L1 no-MLP**: Fixed training is **better** (simpler is better)
For **L2 with MLP** at K=8: Dynamic much better (96% vs 64%)

The optimal training regime depends on model capacity.

### 6. LayerNorm is Critical
**Result**: Removing LayerNorm destroys OOD generalization.
- **With LN**: Generalization reaches 98-100% in best configurations.
- **Without LN**: Generalization is **0.0%** across almost all configurations (tested K17-32), even when training accuracy is reasonable.
- **Implication**: Normalization is essential for learning length-invariant algorithms in this architecture.

## Experiments Completed

### With LayerNorm (~328 runs)
- ✓ 1-2 layers × MLP on/off × mix/curriculum × K=4,6,8,10,12,14,16 × 3 seeds
- ✓ 1-2 layers × MLP on/off × fixed K=4,6,8,10,12,14,16 × 3 seeds

### Without LayerNorm (168 runs)
- ✓ 1-2 layers × MLP on/off × mix/curriculum × K=4,6,8,10,12,14,16 × 3 seeds (84 jobs)
- ✓ 1-2 layers × MLP on/off × fixed K=4,6,8,10,12,14,16 × 3 seeds (84 jobs)

**Key Result:** LayerNorm appears critical for length generalization in this setup. Without it, models struggle to generalize OOD (0% acc on K17-32) even when fitting training data partially.

## How to Use

### Run Training
```bash
# Dynamic training (multi-length) without MLP
sbatch code/run_len_generalization.sh

# Fixed single-K training
sbatch code/run_len_generalization_single_k.sh

# Without layer norm variants
sbatch code/run_len_generalization_no_ln.sh
sbatch code/run_len_generalization_single_k_no_ln.sh
```

### Generate Analysis
```bash
# Comprehensive analysis with all plots and summaries
python code/plot_all_final.py

# Basic plots (older, simpler version)
python code/plot_len_generalization_results.py
```

### Understanding Task Directories

Each experiment group (e.g., `len_gen_k8`) contains **4 task directories** organized by (layers, length_mode or MLP):

**For dynamic training** (e.g., `len_gen_k8/`):
- `task_0000/`: 2 layers, mix — contains **both MLP on and MLP off** checkpoints, 3 seeds each
- `task_0001/`: 2 layers, curriculum — contains **both MLP on and MLP off**, 3 seeds each
- `task_0002/`: 1 layer, mix — contains **both MLP on and MLP off**, 3 seeds each (⭐ MLP off = best)
- `task_0003/`: 1 layer, curriculum — contains **both MLP on and MLP off**, 3 seeds each

**For fixed single-K training** (e.g., `len_gen_single_k4/`):
- `task_0000/`: 2 layers, MLP on (3 seeds)
- `task_0001/`: 2 layers, MLP off (3 seeds)
- `task_0002/`: 1 layer, MLP on (3 seeds)
- `task_0003/`: 1 layer, MLP off (3 seeds) (⭐ ~98% generalization at K=4!)

⭐ To find specific checkpoints, **look at the filename**: `_mlp0_` = no MLP, `_mlp1_` = with MLP

See `checkpoints/TASK_MAPPING.md` for complete details and how to decode task directories.

### Checkpoint Organization
See `checkpoints/README.md` for details on checkpoint naming and organization.

Checkpoints are named:
```
{Prefix}_N128_d128_H1_L{layers}_npos1_mlp{0|1}_ln{0|1}_len{mix|curriculum}_trainK{min}to{max}_testK17to32_nodup1_iters{step}.pt
```

Example:
```
Final_N128_d128_H1_L1_npos1_mlp0_ln1_lenmix_trainK4to4_testK17to32_nodup1_iters40000.pt
```
- L1 = 1 layer
- mlp0 = no MLP (attention-only)
- ln1 = with LayerNorm
- lenmix = uniform mixing data loading
- trainK4to4 = fixed training on K=4 only

## Citation & Related Work

This work builds on insights from:
- "Grokking" phenomena in algorithmic tasks
- Length generalization in transformers
- Curriculum learning and data ordering

The repository is based on [gatmiry/sort-llm](https://github.com/gatmiry/sort-llm).

## Contact

For questions about these experiments:
- Shan Chen (shan.chen@childrens.harvard.edu)
