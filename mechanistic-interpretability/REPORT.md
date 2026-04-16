# SortGPT Mechanistic Interpretability Report

*Last updated: April 15, 2026*

## Model & Setup

- **Architecture**: 2-layer transformer (1 attention head, 64-dim embeddings) with MLP, pre-norm (LayerNorm before attention and MLP)
- **Task**: Sort sequences of integers. Input: `[unsorted | SEP | sorted]`, trained with next-token prediction on the sorted output region.
- **Primary model analyzed**: `k=32` (sequence length), `N=512` (vocabulary size), `init_std=0.01`, `lr=0.03`, `seed=1`
- **Checkpoints**: 60k and 100k iterations (most analysis at 100k)

## Overview of Findings

The model implements a clean, interpretable sorting circuit across its two layers. The key insight is that **Layer 1 MLP is the critical bottleneck** — it processes both the token's own identity and cross-position context from Layer 1 attention to produce the representations that Layer 2's QK matching relies on exclusively.

---

## 1. Layer 2 Attention Mechanism (April 10, 2026)

### 1.1 Consecutive Attention Patterns
**Scripts**: `plot_consecutive_attention.py`, `plot_consecutive_attention_grid.py`
**Plots**: `consecutive_attention_grid.png`, `consecutive_attention_grid_ckpt60k.png`

For inputs containing consecutive numbers i through i+10:
- **Layer 1** (blue) attention at the sorted position of i+1 is spread across several of the consecutive numbers
- **Layer 2** (red) attention is sharply concentrated on **i+2** — the correct next value to predict from that position

### 1.2 Intervention Experiments
**Script**: `plot_intervened_consecutive.py`
**Plots**: `intervened_consecutive.png` (intensity=0), `intervened_consecutive_int5.png` (intensity=5)

Pre-softmax logit interventions on Layer 1 attention:
- Redirecting Layer 1 attention toward different numbers among i..i+10 causes Layer 2 attention to shift accordingly
- At intensity=5, the effect is dramatic: Layer 2 redirects to attend to numbers near the L1 intervention target
- Annotations show that L2 typically puts 70-90% of attention on the consecutive numbers, with the remainder on other unsorted tokens and the sorted output region

---

## 2. QK Score Structure (April 10, 2026)

### 2.1 QK Neighbor & Cross Scores
**Scripts**: `plot_qk_neighbor_cross.py`, `plot_qk_cross_local.py`, `plot_qk_cross_overlay.py`, `plot_qk_cross_mean.py`
**Key plot**: `qk_neighbor_cross.png`

For processed token representations (through Layer 1 ± MLP):
- The **neighbor score** (v_i vs v_{i+1}) is consistently higher than the **self-score** (v_i vs v_i) when MLP is included — creating a monotonic "prefer the next value" signal
- Without MLP, this asymmetry is absent or inconsistent
- The MLP is essential for establishing the monotonic ordering that L2's QK matching relies on

### 2.2 QK Heatmaps and Monotonicity
**Scripts**: `plot_qk_heatmap.py`, `plot_qk_monotonicity_summary.py`

- The full score(i, j) heatmap shows strong asymmetry: scores for j > i are consistently higher than for j < i (with MLP)
- At N=256, this consistent direction reaches 95%+ at larger offsets
- The MLP is confirmed as the critical component for this monotonic structure

---

## 3. Residual Stream Decomposition — Key Finding (April 10, 2026)

### 3.1 Which Residual Components Matter for L2 Attention?
**Script**: `plot_qk_interaction_decomp.py`
**Plot**: `qk_interaction_decomp.png`

The residual stream before Layer 2 attention has three components at each position:
1. **Embed+Pos** (token + positional embedding)
2. **Layer 1 Attention output** (c_proj included)
3. **Layer 1 MLP output**

Applying LN and W_K/W_Q to each component separately and measuring QK scores at the correct key:
- **L1 MLP output dominates**: score = 313, equal to the full residual score of 313
- Embed+Pos: score = -27 (negative!)
- L1 Attention: score = 8 (negligible)

### 3.2 Causal Ablation Study — Definitive Confirmation
**Script**: `plot_qk_deep_decomp.py`
**Plot**: `qk_deep_decomp.png`

Rigorous zero-ablation experiments modifying only key positions (unsorted input), measuring L2 attention accuracy on the correct next-value key (400 trials, 24,800 predictions):

| Condition | L2 attn weight | Top-1 accuracy |
|---|---|---|
| **Normal** | **0.805** | **99.5%** |
| Zero MLP at keys | 0.002 | 7.2% |
| Zero L1 Attn (causal — MLP recomputed) | 0.230 | 70.4% |
| Zero L1 Attn (direct only — MLP kept) | 0.806 | 99.4% |
| MLP from embed only | 0.230 | 70.4% |
| MLP from L1 attn only | 0.005 | 8.0% |
| **MLP only (zero embed + attn in residual)** | **0.805** | **99.5%** |

**Key conclusions**:
- **Removing MLP destroys L2 attention** (99.5% → 7.2%). The MLP output is essential.
- **Removing embed + attn from the residual has ZERO effect** (MLP only → 99.5%). Only the MLP output matters; embed and L1 attn output in the residual are invisible to L2.
- **MLP from embed alone gives 70%**: the token's own identity is the primary MLP input.
- **MLP from L1 attn alone gives 8%**: L1 attention output alone is insufficient.
- **"Zero Attn causal" = "MLP from embed only"** (both 70.4%): confirms the direct residual path of L1 attn is irrelevant — only what flows through the MLP matters.

### 3.3 Query-Side Ablation
**Script**: `plot_qk_query_side_decomp.py`
**Plot**: `qk_query_side_decomp.png`

Same experiments but modifying sorted-output positions (queries) instead of keys:

| Condition | L2 attn weight | Top-1 accuracy |
|---|---|---|
| Normal | 0.803 | 99.5% |
| Zero MLP (queries) | 0.063 | 7.2% |
| MLP from embed only (Q) | 0.215 | **23.1%** |
| MLP from attn only (Q) | 0.045 | 4.5% |
| MLP only (Q side) | 0.803 | 99.5% |

- Same structural story: MLP dominates, direct residual paths irrelevant
- But **L1 attention is much more important for queries** (23% vs 70% for keys with embed-only MLP)
- The query needs contextual information from L1 attention to know "what value to look for next"

---

## 4. When Does L1 Attention Matter? (April 10, 2026)

### 4.1 Detailed Analysis of the 30% Key-Side Gap
**Script**: `plot_attn1_importance_analysis.py`
**Plot**: `attn1_importance_analysis.png`

For the ~30% of predictions where embed-only MLP fails at keys but normal succeeds (800 trials, 24,800 predictions):

**The dominant factor is the gap between consecutive sorted values:**
- Gap = 1: ~45% failure rate (L1 attention needed almost half the time)
- Gap = 2-3: ~33-38% failure rate
- Gap = 11-20: ~14% failure rate
- Gap = 21-50: ~7% failure rate

**Local density is equally predictive:**
- 0 neighbors within ±5: ~15% failure rate
- 2 neighbors within ±5: ~40% failure rate
- 4+ neighbors within ±5: ~60% failure rate

**When embed-only fails, the error is to a nearby number** (median distance = 10). The MLP without L1 attention context encodes "roughly value X" but cannot distinguish X from X±1 when multiple similar values exist.

### 4.2 L1 Attention Patterns at Unsorted Positions
- Mean self-attention: **24%** (each position gives ~1/4 attention to itself)
- Mean cross-attention to other unsorted positions: **76%**
- L1 attention actively moves information between positions, and this cross-position information is what the MLP uses for the 30% accuracy boost

---

## 5. QK Value Vector Heatmaps (April 10, 2026)

### 5.1 Pure L1 Value Vectors (No Base Embedding)
**Script**: `plot_qk_value_heatmap.py`
**Plot**: `qk_value_heatmap.png`

score(x, y) where Q and K are derived purely from L1 value vectors V_x, V_y through MLP1:
- **No "next value" matching pattern**: argmax offset = +1 for 0/512 tokens
- Dominated by horizontal bands (certain tokens universally high/low)
- L1 value vectors alone do not create a sorting-specific QK pattern

### 5.2 With Base Embeddings (z, t)
**Script**: `plot_qk_value_heatmap_base.py`
**Plot**: `qk_value_heatmap_base.png`

Adding token embeddings e_z (query base) and e_t (key base):
- Still dominated by horizontal bands
- Changing t barely affects the heatmap structure
- No diagonal or +1 offset pattern emerges

### 5.3 Decomposed Heatmaps: (t, y), (z, y), and (x, t)
**Scripts**: `plot_qk_heatmap_ty.py`, `plot_qk_heatmap_zy.py`, `plot_qk_heatmap_xt.py`
**Plots**: `qk_heatmap_ty.png`, `qk_heatmap_zy.png`, `qk_heatmap_xt.png`, `qk_heatmap_xt_60k.png`

Varying two of the four variables (z, x, t, y) while fixing the other two:

| Heatmap axes | Variance ratio (dominant/secondary) | Pattern |
|---|---|---|
| **(t, y)** — key base vs key L1 target | t/y = **50-108x** | Horizontal bands (t dominates) |
| **(z, y)** — query base vs key L1 target | z/y = **60-80x** | Horizontal bands (z dominates) |
| **(x, t)** — query L1 target vs key base | t/x = **1.5-2.1x** | Blocky texture, **both axes matter** |

The (x, t) heatmap is the most interesting: the query-side L1 target (x) and the key-side base token (t) have the most comparable influence — the query's contextual information interacts meaningfully with the key's token identity. This is the core of the matching circuit.

---

## 6. Summary: The Sorting Circuit (April 10, 2026)

```
Layer 1 Attention
  └─ At unsorted positions: 76% cross-attention, gathers context about nearby numbers
  └─ At sorted positions: attends to previous sorted outputs + unsorted input
       ↓
Layer 1 MLP (THE CRITICAL BOTTLENECK)
  └─ At key positions: primarily uses token's own embedding (70% alone),
     with L1 attention providing 30% refinement for dense regions
  └─ At query positions: needs BOTH embedding and L1 attention context (23% without L1 attn)
  └─ Produces the ONLY residual component that L2 attention uses
       ↓
Layer 2 Attention
  └─ QK matching relies exclusively on MLP1 output (embed + L1 attn direct paths = 0 effect)
  └─ Achieves 99.5% top-1 accuracy on correct next-value key
  └─ Attention concentrated on the correct next sorted value
```

The raw embeddings and Layer 1 attention outputs, despite being present in the residual stream, are completely invisible to Layer 2's attention mechanism. The entire information flow is funneled through Layer 1's MLP, which acts as the sole communication channel between the layers.

---

## Files

### Analysis Scripts
- `plot_qk_interaction_decomp.py` — QK score decomposition by residual component
- `plot_qk_deep_decomp.py` — Causal ablation study (key-side)
- `plot_qk_query_side_decomp.py` — Causal ablation study (query-side)
- `plot_attn1_importance_analysis.py` — When does L1 attention matter?
- `plot_qk_value_heatmap.py` — Pure L1 value vector QK heatmap
- `plot_qk_value_heatmap_base.py` — QK heatmap with base embeddings
- `plot_qk_heatmap_ty.py` — (t, y) decomposed heatmap
- `plot_qk_heatmap_zy.py` — (z, y) decomposed heatmap
- `plot_qk_heatmap_xt.py` — (x, t) decomposed heatmap
- `plot_intervened_consecutive.py` — Attention intervention experiments
- `plot_consecutive_attention.py` / `plot_consecutive_attention_grid.py` — Consecutive attention patterns
- `plot_qk_neighbor_cross.py` — Neighbor/self QK score analysis
- (and additional earlier scripts for QK cross-scores, monotonicity, etc.)

### Training
- `run_grid.py` — Training orchestrator for grid of models across 8 GPUs
- `plot_checkpoint_analysis.py` — Standard checkpoint analysis plots

---

## 7. Attn2 Dependence on Token Value: The Small-i Puzzle (April 14–15, 2026)

**Model**: k32_N512, seed 1, 100k checkpoint (two-mode model where L2 attention is critical)

### 7.1 Core Observation

Removing attn2 (while keeping MLP2, LayerNorm, etc.) causes per-token accuracy to depend dramatically on the **numerical value** `i` of the current sorted token, not its position in the sequence:

| Value range | Accuracy without attn2 (gap=1) |
|---|---|
| i ∈ [0, 80) | ~6% |
| i ∈ [80, 180) | ~40–80% (transition zone) |
| i ∈ [180, 350) | ~95% |
| i ∈ [350, 512) | ~99% |

**Plots**: `no_a2_acc_by_i_gap1.png`, `no_a2_acc_by_i_multigap.png`, `no_a2_acc_by_i_largegap.png`

This pattern holds across all gaps, not just gap=1. Larger gaps shift the transition zone rightward — the model needs larger `i` before attn2 becomes dispensable.

**Plot**: `no_a2_acc_and_a1_weight_by_i_multigap.png`

### 7.2 The Puzzle: L1 Is Better for Small i

The natural hypothesis — that L1 attention fails more often for small `i`, making attn2 necessary — is **wrong**. Measuring L1 attention accuracy (whether the top-attended unsorted key is the correct target):

| Value range | L1 error rate | Avg attn on target |
|---|---|---|
| i ∈ [0, 50) | **2.1%** | 0.887 |
| i ∈ [50, 100) | 3.9% | 0.839 |
| i ∈ [100, 200) | 3.5% | 0.823 |
| i ∈ [200, 300) | 4.9% | 0.802 |
| i ∈ [300, 400) | **6.7%** | 0.772 |
| i ∈ [400, 512) | 4.0% | 0.803 |

L1 attention is **more accurate and sharper** for small `i` than for large `i`. Yet removing attn2 is catastrophic for small `i` and harmless for large `i`. The information entering the residual stream from L1 is at least as good for small `i` — the problem is downstream.

When L1 does make errors (in either range), the mistakes are nearly identical in character: it picks a value off by +1 or +2 from the correct target.

**Plot**: `no_a2_acc_and_a1_weight_by_i_gap1.png`

### 7.3 Hypotheses Tested and Ruled Out

**Hypothesis 1: Embedding space geometry.** Consecutive embeddings `wte(i)` and `wte(i+1)` have lower cosine similarity for small `i` (~0.50) than large `i` (~0.85), and small numbers have more "impostors" (numerically distant tokens that are embedding-close). This could explain why the tied `lm_head = wte` readout is harder for small numbers.

**Counter-argument (from observation):** The model can learn arbitrary projections through its value weights `W_V`, output projection `W_O`, and MLP weights. The same way it learns `W_Q`/`W_K` that create clean geometry for attention lookup despite messy raw embeddings, it could learn OV-circuit and MLP weights that align well with the `lm_head` readout for all `i` values. The embedding geometry is not a fundamental constraint.

**Hypothesis 2: SEP token attention.** Perhaps L1 attention "wastes" probability mass on the SEP token for small `i`.

**Ruled out:** SEP attention is negligible across all `i` values (mean 0.0006, max 0.007). It shows no dependence on `i`.

**Hypothesis 3: L1 attention strength on target correlates with attn2 dependence.**

**Partially ruled out:** L1 attention on the correct target is ~0.89 for small `i` and ~0.80 for large `i` — actually higher for small `i`. The correlation between L1 attention weight and no-attn2 accuracy, when controlling for `i`, shows a non-monotonic pattern confounded by `i` value.

**Plot**: `a1_strength_vs_no_a2_acc_ci.png`

### 7.4 What We Do Know

**Attn2 changes MLP2's operating mode.** The cosine similarity between MLP2's input (`ln_2(residual)`) with vs. without attn2 is:
- Small `i`: **0.36** — MLP2 sees a completely different vector
- Large `i`: **0.75** — MLP2 sees a similar vector

MLP2 produces fundamentally different output depending on whether attn2 is present, and this difference is much more dramatic for small `i`.

**The effect is gradual, not binary.** Scaling attn2's output by a factor α:
- Small `i`: needs α ≥ 0.7 for 100% accuracy (6% at α=0, 48% at α=0.2, 76% at α=0.3)
- Transition `i`: needs α ≥ 0.3 for ~100%
- Large `i`: 99%+ at α=0

**Plot**: `attn2_scaling_by_i_range.png`

**Attn2's output doesn't encode the answer directly.** The cosine similarity between attn2's output vector and `wte(target)` is ~0.00 for both small and large `i`. Attn2 is not simply "writing the target embedding into the residual stream."

**When the model fails at small `i` without attn2**, it predicts numbers that are off by a moderate amount (median offset +9 from target) — it's in the right neighborhood but lacks precision.

**Logit lens shows no stage before MLP2 resolves the answer.** The correct token rank at each residual stream stage (for small `i`):
- After embed: rank ~13
- After attn1: rank ~83
- After MLP1: rank ~207 (worse!)
- After attn2: rank ~122
- After MLP2: rank 0

For large `i`, the progression is similar but MLP2 can resolve from rank ~123 (without attn2) to rank 0.

**Embedding geometry**:
- `wte(5)`'s nearest neighbors: [6, **437**, 4, **35**, **36**] — chaotic
- `wte(500)`'s nearest neighbors: [501, 499, 502, 498, 511] — perfectly organized

**Plot**: `embedding_geometry_vs_a2_dependence.png`

### 7.5 What Remains Unknown

The central puzzle is: **why does the model's MLP pathway (MLP1 → MLP2) fail to produce the correct output for small `i` without attn2, even though L1 provides better information for small `i` than for large `i`?**

The model has sufficient capacity (64-dim embeddings, learnable projections at every stage) to potentially route information correctly for all `i` values. The fact that it distributes computation such that attn2 is needed only for small `i` is a **training outcome** — the gradient descent optimization found this division of labor, but we have not established whether it is a necessary consequence of the architecture or merely a convenient local minimum.

Possible directions:
1. The MLP's finite width may create capacity trade-offs where optimizing for the large-`i` majority (more common in training) leaves small-`i` cases under-served by the L1+MLP1 path alone.
2. The interaction between the specific nonlinearities in MLP1/MLP2 and the tied lm_head readout may create representational bottlenecks that are harder to satisfy for small numbers.
3. Training dynamics: early in training, the model may learn a general attn2-dependent strategy, and then later develop the L1-sufficient shortcut only for the "easier" large-`i` cases.

### 7.6 Summary

| Property | Small i (0–80) | Large i (300+) |
|---|---|---|
| L1 attention error rate | 2.1% (better) | 6.7% (worse) |
| L1 attention on target | 0.89 (higher) | 0.77 (lower) |
| Accuracy without attn2 | ~6% (fails) | ~99% (works) |
| MLP2 input change from attn2 | cos=0.36 (massive) | cos=0.75 (modest) |
| Attn2 output norm | 25,400 (large) | 13,500 (smaller) |
| Embedding organization | chaotic | well-organized |

The model receives better first-layer information for small numbers but is more dependent on second-layer attention — an apparent paradox whose resolution likely lies in the interaction between MLP capacity allocation, training dynamics, and the readout through tied embeddings.

### 7.7 Plots

All plots are in `sort-llm/new-grid/k32_N512/plots/`:

- `no_a2_acc_by_i_gap1.png` — Accuracy without attn2 vs i (gap=1)
- `no_a2_acc_by_i_multigap.png` — Same for gaps 1, 2, 3, 5, 10
- `no_a2_acc_by_i_largegap.png` — Same for gaps 16, 20, 30, 50
- `no_a2_acc_and_a1_weight_by_i_gap1.png` — Dual-axis: accuracy + L1 attention weight vs i (gap=1)
- `no_a2_acc_and_a1_weight_by_i_multigap.png` — Same for multiple gaps
- `a1_strength_vs_no_a2_acc_ci.png` — Accuracy vs L1 attention strength with confidence intervals
- `attn2_scaling_by_i_range.png` — Effect of scaling attn2 output by α for small/transition/large i
- `embedding_geometry_vs_a2_dependence.png` — Embedding organization metrics vs attn2 dependence
- `attn2_mechanism_summary.png` — Multi-panel summary figure
