# Residual Stream Cleanup: Redundant Components in Leap-Former Models

## Overview

This study tests whether two residual stream components carry no useful information
for the final prediction in leap-former models — i.e., whether they can be removed
from the residual without changing the model's argmax output.

**Claim 1 (attn1 direct path is redundant):**
The output of layer-1 attention (`attn1_out`) can be removed from the residual
stream *after* it has served as input to MLP1. Concretely, the residual entering
block 1 becomes `embed + mlp1_out` instead of `embed + attn1_out + mlp1_out`.
The attn1 information that matters is already captured inside `mlp1_out` (which
was computed from `embed + attn1_out`); the direct additive contribution of
`attn1_out` itself is redundant.

**Claim 2 (mlp1 direct path to readout is redundant):**
The output of MLP1 (`mlp1_out`) can be removed from the final residual before
`ln_f` + `lm_head`, as long as it is present for the computation of attn2 and
MLP2. The information mlp1 carries is consumed by attn2 (for QK matching) and
MLP2, but has no direct role in the final readout.

## Model Grid

- **Architectures**: 2-layer, 1-head, 64-dim transformers with MLP and pre-norm
- **Configs**: k ∈ {16, 32} × N ∈ {128, 256, 512, 1024}, 5 seeds each = 40 models
- **Checkpoints**: final (100k iterations) from `new-grid/` (seed 1) and `new-grid-multiple/` (seeds 2–5)
- **Leap-former classification**: per-token accuracy with attn2 ablated < 90% → 17 out of 40

## Results

### Claim 1: Universal

Across all 17 leap-formers, removing `attn1_out` after it fed MLP1 preserves
the argmax prediction at **98.5–100%** of positions (mean 99.87%).
This holds without exception.

### Claim 2: Holds for 12/17, fails for 5

| Group | Models | Claim 2 agreement |
|-------|--------|-------------------|
| k32_N512 (all 5 seeds) | 5 | 96.6–100.0% |
| k32_N256 (seeds 1,2,3) | 3 | 91.3–98.7% |
| k16_N128_s4, k16_N512_s2, k16_N1024_s2, k16_N1024_s5 | 4 | 96.2–100.0% |
| **k16_N1024_s1, s3, s4** | 3 | **46.6–47.2%** |
| **k16_N512_s5** | 1 | **37.5%** |
| **k32_N1024_s5** (known outlier) | 1 | **2.4%** |

The 5 models where Claim 2 fails include the two previously identified outliers
(`k16_N1024_s3`, `k32_N1024_s5`) where attn2 depends on more than just MLP1
output, plus three additional k16 models.

For the 12 non-failing leap-formers, both claims hold at >91% agreement.

## Reproduction

### Prerequisites
- Python 3, PyTorch, NumPy, matplotlib
- Checkpoint files in `new-grid/` and `new-grid-multiple/`

### Run the analysis
```bash
cd sort-llm/mechanistic-interpretability/residual-stream-cleanup/
python test_residual_removal_all.py
```

This produces `residual_removal_results.json` with per-checkpoint data.

### Generate the figure
```bash
python plot_residual_removal.py
```

Output: `plots/residual_stream_cleanup.png`

## Files

| File | Description |
|------|-------------|
| `test_residual_removal_all.py` | Main analysis script (classifies leap-formers, tests both claims) |
| `plot_residual_removal.py` | Generates the publication figure |
| `residual_removal_results.json` | Raw results for all 40 checkpoints |
| `plots/residual_stream_cleanup.png` | Publication figure |
