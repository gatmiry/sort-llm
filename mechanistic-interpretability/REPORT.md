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

The model receives better first-layer information for small numbers but is more dependent on second-layer attention — an apparent paradox whose resolution is detailed in Section 8.

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

---

## 8. Information Flow and Dual-Role Architecture (April 15–16, 2026)

**Model**: k32_N512, seed 1, 100k checkpoint (`std0p01_iseed1__ckpt100000.pt`)
**All experiments below use gap=1 cases** (current number `i`, correct next = `i+1`) **unless stated otherwise.**

### 8.1 Which Residual Stream Components Reach the Output?

We tested which components of the residual stream are actually needed at the final readout (`ln_f → lm_head`) and as input to MLP2. The full residual stream is: `embed + attn1_out + mlp1_out + attn2_out + mlp2_out`.

| Ablation | Small i (0–80) | Transition (150–250) | Large i (350–512) |
|---|---|---|---|
| Normal (no ablation) | 100% | 100% | 100% |
| Remove `attn1_out` from final readout only | 100% | 100% | 100% |
| Remove `attn1_out` from MLP2 input + final readout | 100% | 100% | 100% |
| Remove `mlp1_out` from final readout only | 100% | 100% | ~100% |
| **Remove `mlp1_out` from MLP2 input + final readout** | **0.0%** | **0.1%** | **0.6%** |

**Key conclusions:**
- **`attn1_out` is invisible** to both MLP2 and the final readout. It can be removed from anywhere downstream without effect. Its sole purpose is as input to MLP1.
- **`mlp1_out` is invisible to the final readout** but **critically needed by MLP2**. Without `mlp1_out` in its input, MLP2 is completely non-functional (0% accuracy).
- The effective final readout depends only on: `embed + attn2_out + mlp2_out`. The first layer's direct contributions (`attn1_out`, `mlp1_out`) flow entirely through downstream components.

**Information flow summary:**
```
attn1 → [message to MLP1] → mlp1 → [message to MLP2 and attn2] → attn2 + mlp2 → readout
                                                                         ↓
                                                              embed + attn2_out + mlp2_out → ln_f → lm_head
```

### 8.2 Scaling MLP1's Contribution to MLP2

Instead of binary removal, we scaled `mlp1_out` in MLP2's input by α ∈ [0, 1] (MLP2 input: `ln_2(embed + α·mlp1_out + attn2_out)`). This reveals how sensitive MLP2 is to the magnitude of `mlp1_out`.

All `i` ranges show a **sharp sigmoid-like transition** from 0% to 100% accuracy, but with different thresholds:
- **Mid-high i (250–350)**: recovers fastest (starts at α≈0.15, saturates by α≈0.55)
- **Transition i (150–250)**: next (saturates by α≈0.55)
- **Large i (350–512)**: slower (needs α≈0.8 for full recovery)
- **Small i (0–80)**: slowest (near 0% until α≈0.4, needs α≈0.75)

**Plot**: `mlp1_scaling_in_mlp2_gap1.png`

MLP2 has learned to expect a specific magnitude/structure from `mlp1_out`. Below a threshold α, MLP2 cannot extract the number information.

### 8.3 Hijack Experiments: Which Channel Controls the Output?

We designed three hijack interventions to determine whether `mlp1` or `attn2` controls MLP2's output at different `i` values. In each, we force one channel to point to a wrong number (`i + offset`) while leaving the other channel normal, and measure whether the output follows the hijacked signal.

**MLP1 hijack**: Force attn1 to attend to `i+offset`, recompute `mlp1` from that forced attn1, feed the fake `mlp1` to MLP2 (attn2 computed normally from the real residual).

**ATTN2 hijack**: Force attn2 to attend to `i+offset` in the unsorted sequence (mlp1 stays normal).

**ATTN1 hijack**: Force attn1 to attend to `i+offset`, let everything flow naturally (both mlp1 and attn2 see the hijacked info).

#### Results at offset +5 (gap=1):

| i range | MLP1 hijack | ATTN2 hijack | ATTN1 hijack |
|---|---|---|---|
| 0–20 | **100%** | 1.1% | 95.9% |
| 20–80 | 63.8% | 21.3% | 61.3% |
| 80–200 | 64.1% | 37.0% | 71.4% |
| 200–350 | 67.6% | 48.8% | 67.4% |
| 350–512 | 82.4% | **67.2%** | 79.5% |

**MLP1 hijack and ATTN2 hijack are mirror images:**

- **Small i (0–20)**: MLP1 hijack succeeds at ~100%, ATTN2 hijack at ~1%. → **mlp1 controls the output; attn2 cannot override it.**
- **Large i (350–512)**: ATTN2 hijack succeeds at 67–99% (offset-dependent), MLP1 hijack at 68–82%. → **Both channels carry number info; attn2 has strong influence.**

This pattern holds across all tested offsets (+2, +3, +5, +10, +20):

For small i (0–20):
| Offset | MLP1 hijack | ATTN2 hijack |
|---|---|---|
| +2 | 82.5% | 36.7% |
| +3 | 96.1% | 11.0% |
| +5 | 100% | 1.1% |
| +10 | 95.1% | 0.0% |
| +20 | 78.7% | 0.0% |

For large i (350–512):
| Offset | MLP1 hijack | ATTN2 hijack |
|---|---|---|
| +2 | 68.4% | 99.4% |
| +3 | 73.0% | 94.6% |
| +5 | 82.4% | 67.2% |
| +10 | 67.1% | 35.1% |
| +20 | 25.1% | 26.7% |

**ATTN1 hijack** produces results similar to MLP1 hijack but slightly weaker. This is because in ATTN1 hijack, attn2's query is computed from the modified (out-of-distribution) residual, causing attn2 to produce noisy output that interferes with the hijack. When the ATTN1 hijack fails, the model predicts a third number (neither the target nor the original), confirming attn2 produces noise rather than a meaningful correction.

**Plot**: `hijack_comparison_3way.png`

### 8.4 Resolving the Paradox: Enabler vs. Specifier

The hijack results resolve the apparent contradiction between "removing attn2 hurts small i" and "mlp1 controls the output for small i":

**For small i, attn2 is an "enabler," not a "specifier."**
- Its role is to put MLP2 into its operating regime (like a bias term), not to specify which number to output.
- Removing attn2 → 0% accuracy (MLP2 loses the enabling signal and cannot function)
- Keeping attn2 + correct mlp1 → 100% (MLP2 enabled, follows mlp1)
- Keeping attn2 + hijacked mlp1 → ~100% hijacked (MLP2 enabled, still follows mlp1)
- Hijacking attn2 → ~0% success (mlp1 overrides attn2's number signal)

**For large i, attn2 is a "specifier" with authoritative override.**
- When present, MLP2 trusts attn2 and follows it (hijacking attn2 works at 67–99%).
- When absent (removed), MLP2 falls back on mlp1 alone and still works (99.5% accuracy).
- Both mlp1 and attn2 independently carry the correct number information — there is **redundancy**.

**Distinguishing removal from hijacking** is critical: removing attn2 eliminates a signal (MLP2 adapts to its absence for large i). Hijacking attn2 injects a *wrong* signal that actively competes with mlp1 and can win.

Evidence from L2 norms:

| i range | ‖attn2‖ | ‖mlp1‖ | Ratio |
|---|---|---|---|
| 0–20 | 25,438 | 42,773 | 0.595 |
| 350–512 | 12,672 | 42,770 | 0.296 |

Despite attn2 having a *larger* norm for small i, it cannot hijack the output — confirming its role is directionally generic (enabling), not number-specific.

### 8.5 Gap Dependence: The MLP1-Dominance Is Gap-1 Specific

The dual-role picture changes dramatically with gap size:

| Gap | MLP1 hijack (small i) | ATTN2 hijack (small i) | Dominant channel |
|---|---|---|---|
| 1 | **80.2%** | 14.4% | mlp1 |
| 3 | 50.8% | 35.2% | transitioning |
| 5 | 12.2% | **57.5%** | attn2 |
| 10 | 3.0% | **89.8%** | attn2 |
| 20 | 0.6% | **100%** | attn2 |

For gap ≥ 5, **attn2 dominates the output for ALL i ranges**, including small i. By gap=20, MLP1 hijack is 0% and ATTN2 hijack is 100% everywhere.

Correspondingly, accuracy without attn2 drops for large i as gap increases:

| Gap | Large i (250–512) no-attn2 accuracy |
|---|---|
| 1 | 98.9% |
| 3 | 95.4% |
| 5 | 88.9% |
| 10 | 70.3% |
| 20 | 47.3% |

**Interpretation**: For gap=1, mlp1 can learn the simple "i → i+1" increment function directly in MLP1's weights. For larger gaps, the correct next number depends on which specific numbers are in the sequence — information that requires looking at the keys via attn2. The MLP pathway alone cannot determine the answer.

### 8.6 Summary: Two Operating Regimes

The model operates in two regimes determined by **gap size** and **i value**:

| Regime | When | mlp1 role | attn2 role | No-attn2 accuracy |
|---|---|---|---|---|
| **MLP1-driven** | Gap=1, large i | Number specifier | Authoritative override (redundant) | ~99% |
| **MLP1-driven + enabled** | Gap=1, small i | Number specifier | Enabler (needed but doesn't specify) | ~0% |
| **ATTN2-driven** | Gap ≥ 5, all i | Infrastructure | Number specifier | Decreasing with gap |

The transition from MLP1-driven to ATTN2-driven is continuous, occurring around gap=3–5.

### 8.7 Plots

All plots in `sort-llm/new-grid/k32_N512/plots/`:

- `mlp1_scaling_in_mlp2_gap1.png` — Accuracy vs α (scaling mlp1 in MLP2 input), by i range
- `hijack_comparison_3way.png` — Side-by-side: MLP1 hijack vs ATTN1 hijack vs ATTN2 hijack success rates

---

## 9. Attn2 QK Mechanism: Windowed Monotonicity (April 15–16, 2026)

**Model**: k32_N512, seed 1, 100k checkpoint. Representative of all non-outlier leap-formers.

### 9.1 Attn2 Scores Depend Only on MLP1 Output

Verified across 13 leap-former checkpoints (grid of k∈{16,32} × N∈{128,256,512,1024}, 5 seeds each; 2 N=1024 outliers excluded):

- **Distributional similarity**: ℓ₁ distance between full-residual and MLP1-only attn2 distributions is <0.03 for all 13 checkpoints. For k32_N512 seeds, d ≈ 0.005.
- **Accuracy preservation**: Replacing attn2's input with MLP1-only gives 0% accuracy drop in 12/13 checkpoints (max 0.05%).
- **Outliers**: 2 N=1024 leap-formers deviate (d=0.133, 0.299), seed-dependent.

**Scripts**: `attn2-mlp1-dependence/compute_probl1distance.py`
**LaTeX**: `attn2-mlp1-dependence/paper-addon.tex`, `attn2-mlp1-dependence/report.tex`

### 9.2 Asymmetric Role of Attn1 Context

The synthetic score `s(z, x, t, y) = q(z,x) · k(t,y)` reveals:
- **Key-side (y)**: Score nearly invariant to y.
- **Query-side (x)**: Rich structure — high-score band in [z, x].

Attn2 uses attn1 context exclusively on the query side.

### 9.3 Windowed Monotonicity

Slices at x ∈ {260, 270, 280, 290, 300} (z=250):
1. **Band formation**: Score elevated in [z, x], suppressed outside.
2. **Monotonicity**: Approximately monotonic within band; argmax slightly below x.

Explains the 28× error rate improvement of attn2 over attn1.

### 9.4 Bounded Trust (Argmax Saturation)

Argmax tracks x but saturates at moderate distance above z (≈300 for z=250). The model ignores implausibly large attn1 targets.

### 9.5 Smoothness and Tighter Attention

At threshold 0.04: attn1 attends to 1.5–3 keys (distance 2–14); attn2 to 1.2–2.2 keys (distance 0.3–3).

### 9.6 Scripts and Plots

- `attn2-qk-mechanism/generate_all_plots.py` — Generates all 6 mechanism figures
- `attn2-qk-mechanism/REPRODUCTION.md` — Reproduction guide
- `attn2-qk-mechanism/paper-addon.tex` — LaTeX for Overleaf

Figures (in Overleaf `newpics/`): `qk_heatmap_asymmetry.png`, `qk_score_slices_band.png`, `argmax_saturation.png`, `l1_vs_l2_qk_smoothness.png`, `attn_spread_comparison.png`, `attn_error_rates.png`

---

## 10. Split Attn1 Attention and Argmax Bias (April 16–17, 2026)

### 10.1 Robustness to Split Attention

Replacing single-token attn1 output with uniform split: `V_split(x, n, δ) = (1/n) Σ V^(1)_{x+iδ}`. Tested n∈{2,3,4}, δ∈{1,5,20}:

- **Heatmaps**: Band structure preserved for all n.
- **Score slices**: Windowed monotonicity persists. Argmax shifts slightly rightward with larger δ.

### 10.2 Argmax Bias Toward x_min

The argmax t* tracks x_min (smallest token in the attended set), not x_mean or x_max. |t* − x_min| ≪ |t* − x_mean| for n > 1. Functionally advantageous: correct next output depends on smallest candidate above z.

### 10.3 Scripts and Plots

- `attn2-qk-appendix/generate_plots.py` — Heatmaps and slice plots for n=2,3,4
- `attn2-qk-appendix/plot_argmax_bias.py` — Argmax bias figure
- `attn2-qk-appendix/appendix.tex` — LaTeX appendix content

Figures (in `attn2-qk-appendix/plots/`): `qk_heatmap_split_comparison.png`, `qk_slices_split_{2,3,4}tokens.png`, `argmax_bias_analysis.png`

---

## 11. Overleaf Paper Status (April 17, 2026)

Repository: `/mnt/task_runtime/69c9a928b8ca815361b30519/`

### Mechanistic Analysis Sections Pushed (§4)
1. §4.1 Attn2 depends only on MLP1 output (across 13 leap-formers)
2. §4.2 Windowed monotonicity mechanism (asymmetry, band, saturation, smoothness, spread)
3. §4.3 Leap-former sorting circuit summary

### Appendix Sections Pushed
- Appendix A: Initialization scale sweep
- Appendix B: Split attn1 attention
  - B.1 Heatmaps (n=1..4, δ=5)
  - B.2 Score Slices (n=2,3,4 at δ=1,5,20)
  - B.3 Argmax Bias Toward x_min
