# Role of Position: Accuracy Without Attn2 by Current Value and Gap

*Last updated: April 15, 2026*

## Overview

This study measures how well a leap-former model predicts the next sorted token
**without** the second-layer attention (attn2).  We ablate attn2 (skip
block-1 attention, retain MLP2) and record per-token accuracy as a function
of two variables:

| Variable | Definition |
|----------|-----------|
| **Current value** $i$ | The token at the current sorted position |
| **Gap** $g$ | $v_{\text{next}} - i$: the difference to the correct next token |

## Key Findings

1. **Small gaps, large $i$**: When $g \le 2$ and $i \gtrsim 200$, the model
   achieves near-perfect accuracy even without attn2, meaning the first-layer
   pathway alone suffices.
2. **Monotonic degradation with gap**: As $g$ increases, accuracy drops
   systematically — at $g = 50$ the model barely exceeds chance.
3. **Transition region around $i \approx 150$**: For all gap sizes there is a
   sharp accuracy ramp near $i = 150$, suggesting a regime boundary in the
   learned representations.
4. **Boundary effects**: Near the vocabulary boundary ($i \to 512$), accuracy
   degrades slightly for moderate gaps due to edge effects in the token
   distribution.

## Model & Data

- **Checkpoint**: `new-grid/k32_N512/checkpoints/std0p01_iseed1__ckpt100000.pt`
  - $k = 32$ (block size), $N = 512$ (vocabulary), seed 1
- **Number of batches**: 12,000 random batches (each with $k - 1 = 31$ sorted
  prediction positions → 372,000 total position records)
- **Gaps evaluated**: $g \in \{1, 2, 3, 5, 10, 20, 30, 50\}$
- **Smoothing**: Rolling window of 20 values of $i$, with 1,000-sample
  bootstrap for 95% confidence intervals

## Ablation Protocol

For each batch:
1. Generate a random input `[unsorted | SEP | sorted]`
2. Run block 0 (attn1 + MLP1) normally
3. **Skip block-1 attention entirely** (set attn2 output to zero)
4. Run block-1 MLP, final LayerNorm, and language head normally
5. Record per-position argmax predictions vs. ground-truth targets

This matches `compute_ablation(model, block_size, vocab_n, device, skip_layer=1)` from `sortgpt_toolkit/intervene.py`.

## Reproduction

```bash
cd /path/to/sort-llm
python mechanistic-interpretability/role-of-position/plot_no_attn2_acc_by_gap.py
```

Output: `mechanistic-interpretability/role-of-position/plots/no_attn2_acc_by_gap.png`

Requirements: PyTorch, NumPy, Matplotlib, and the `sortgpt_toolkit` package
(available in the repository).

## Files

| File | Description |
|------|-------------|
| `plot_no_attn2_acc_by_gap.py` | Main script: data collection, smoothing, plotting |
| `plots/no_attn2_acc_by_gap.png` | Output figure |
| `report.md` | This file |
| `paper-addon.tex` | LaTeX subsection for the paper |
