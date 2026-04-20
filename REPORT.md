# SortGPT Mechanistic Interpretability Report

## Model & Setup

- **Architecture**: 2-layer transformer (1 attention head, 64-dim embeddings) with MLP, pre-norm (LayerNorm before attention and MLP)
- **Task**: Sort sequences of integers. Input: `[unsorted | SEP | sorted]`, trained with next-token prediction on the sorted output region.
- **Primary model analyzed**: `k=32` (sequence length), `N=512` (vocabulary size), `init_std=0.01`, `lr=0.03`, `seed=1`
- **Checkpoints**: 60k and 100k iterations (most analysis at 100k)

## Overview of Findings

The model implements a clean, interpretable sorting circuit across its two layers. The key insight is that **Layer 1 MLP is the critical bottleneck** — it processes both the token's own identity and cross-position context from Layer 1 attention to produce the representations that Layer 2's QK matching relies on exclusively.

---

## 1. Layer 2 Attention Mechanism

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

## 2. QK Score Structure

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

## 3. Residual Stream Decomposition (Key Finding)

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

## 4. When Does L1 Attention Matter?

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

## 5. QK Value Vector Heatmaps

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

## 6. Summary: The Sorting Circuit

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

## 7. Attn2 QK Mechanism: Windowed Monotonicity (April 15–16, 2026)

**Model**: k32_N512, seed 1, 100k checkpoint. Representative of all non-outlier leap-formers.

### 7.1 Attn2 Scores Depend Only on MLP1 Output

Verified across 13 leap-former checkpoints (grid of k∈{16,32} × N∈{128,256,512,1024}, 5 seeds each; 2 N=1024 outliers excluded):

- **Distributional similarity**: ℓ₁ distance between full-residual and MLP1-only attn2 distributions is <0.03 for all 13 checkpoints. For k32_N512 seeds, d ≈ 0.005.
- **Accuracy preservation**: Replacing attn2's input with MLP1-only gives 0% accuracy drop in 12/13 checkpoints (max 0.05%).
- **Outliers**: 2 N=1024 leap-formers deviate (d=0.133, 0.299), showing attn2 can exploit other residual components.

**Scripts**: `mechanistic-interpretability/attn2-mlp1-dependence/compute_probl1distance.py`

### 7.2 Asymmetric Role of Attn1 Context

The score function `s(z, x, t, y) = q(z,x) · k(t,y)` reveals:
- **Key-side (y)**: Score is nearly invariant — changing attn1's target on the key side has no effect.
- **Query-side (x)**: Rich structure. A high-score band emerges in [z, x] as x increases.

Attn2 uses contextual information from attn1 exclusively on the query side.

### 7.3 Windowed Monotonicity

Slices at x ∈ {260, 270, 280, 290, 300} (z=250):
1. **Band formation**: Score elevated in window [z, x], suppressed outside.
2. **Monotonicity**: Approximately monotonic within the band; argmax lies slightly below x.

This explains how attn2 (99.5%) vastly outperforms attn1 (86%): monotonicity is only needed within a narrow window.

### 7.4 Bounded Trust (Argmax Saturation)

The argmax tracks x initially but saturates at a moderate distance above z. For z=250, saturation at t≈300 regardless of x. The model ignores implausibly large attn1 targets.

### 7.5 Smoothness Advantage and Tighter Attention

Attn2 QK scores are smoother than attn1 within their windows. At threshold 0.04: attn1 attends to 1.5–3 keys (distance 2–14), attn2 to 1.2–2.2 keys (distance 0.3–3).

### 7.6 Scripts and Plots

- `mechanistic-interpretability/attn2-qk-mechanism/generate_all_plots.py` — All 6 mechanism figures
- `mechanistic-interpretability/attn2-qk-mechanism/REPRODUCTION.md` — Detailed reproduction guide
- Plots in Overleaf `newpics/`: `qk_heatmap_asymmetry.png`, `qk_score_slices_band.png`, `argmax_saturation.png`, `l1_vs_l2_qk_smoothness.png`, `attn_spread_comparison.png`, `attn_error_rates.png`

---

## 8. Hijack Experiments: MLP1 vs ATTN2 Contributions (April 18–20, 2026)

**Scripts**: `mechanistic-interpretability/role-of-position/plot_hijack_per_i.py`, `plot_no_attn2_by_group.py`, `plot_hijack_avg_seeds.py`
**Models**: k32_N512, seeds 1–5, 100k checkpoint

### 8.1 Experimental Design

Four hijack conditions measured per (current_value i, offset) pair:
1. **MLP1 hijack**: Force attn1 to attend to i+offset, recompute mlp1, feed modified mlp1 to normal attn2/mlp2
2. **ATTN2 hijack**: Force attn2 to attend to i+offset, mlp1 stays normal
3. **Both simultaneously**: Apply both MLP1 and ATTN2 hijacks to the same wrong target
4. **Both individually succeed**: Fraction where independent MLP1 AND ATTN2 hijacks each predict the hijacked value

### 8.2 Gap=1: Two Distinct Learned Strategies

**Seed 1** (hybrid strategy):
- Small i (1–3): MLP1 hijack ~100%, ATTN2 hijack ~0% → MLP1 specifies the number
- Large i (477–497): ATTN2 hijack dominates, MLP1 partially effective → Dual pathway
- Smooth monotonic transition across vocabulary

**Seed 4** (MLP1-dominant strategy):
- MLP1 hijack ~100% across ALL i values (1 through 497)
- ATTN2 hijack ~0% everywhere
- The model routes everything through MLP1 for gap=1

This demonstrates that multiple valid sorting circuits exist under identical training configurations — only the random seed differs.

### 8.3 Larger Gaps (10, 20, 40): Universal ATTN2 Dominance

For gap ≥ 10, all seeds converge to ATTN2-dominant behavior:
- MLP1 hijack success drops sharply
- ATTN2 hijack success approaches 100%
- This holds across all i-values and all seeds

### 8.4 Attn2 Ablation Confirms Hijack Results

Removing attn2 (keeping MLP2) and measuring per-token accuracy:
- **Seed 1**: Strong i-dependence — small i catastrophic (~6%), large i fine (~99%)
- **Seed 4**: Much less i-dependent — MLP1 alone handles most gap=1 cases
- Both seeds show attn2 is critical for large gaps regardless of i

### 8.5 Cross-Seed Averaging

All 5 seeds (seed 1 from new-grid, seeds 2–5 from new-grid-multiple) run with averaged-over-all-i data for gaps 1, 10, 20, 40. Combined plot shows mean ± std across seeds, revealing which behaviors are universal vs seed-specific.

---

## 9. Split Attn1 Attention and Argmax Bias (April 16–17, 2026)

### 9.1 Robustness to Split Attention

Replacing single-token attn1 output with uniform split: `V_split(x, n, δ) = (1/n) Σ V^(1)_{x+iδ}`. Tested n∈{2,3,4}, δ∈{1,5,20}:

- **Heatmaps**: Band structure preserved for all n.
- **Score slices**: Windowed monotonicity persists. Argmax shifts slightly rightward with larger δ but remains modest even at n=4, δ=20 (60-token span).

### 9.2 Argmax Bias Toward x_min

When attn1 splits across n tokens, the argmax t* tracks x_min (smallest token), not x_mean or x_max. |t* − x_min| ≪ |t* − x_mean| for n > 1.

**Functional advantage**: The correct next output depends on the smallest candidate above z, so anchoring near x_min preserves correct sorting.

### 9.3 Scripts and Plots

- `mechanistic-interpretability/attn2-qk-appendix/generate_plots.py` — Heatmaps and slice plots
- `mechanistic-interpretability/attn2-qk-appendix/plot_argmax_bias.py` — Argmax bias figure
- `mechanistic-interpretability/attn2-qk-appendix/appendix.tex` — LaTeX appendix
- Plots in `mechanistic-interpretability/attn2-qk-appendix/plots/`: `qk_heatmap_split_comparison.png`, `qk_slices_split_{2,3,4}tokens.png`, `argmax_bias_analysis.png`

---

## 10. Overleaf Paper (April 15–17, 2026)

Maintained at `/mnt/task_runtime/69c9a928b8ca815361b30519/` (Overleaf git repo).

### Mechanistic Analysis Sections Pushed (§4 in paper)
1. §4.1 Attn2 depends only on MLP1 output (13 leap-formers)
2. §4.2 Windowed monotonicity mechanism (asymmetry, band formation, saturation, smoothness, spread)
3. §4.3 Leap-former sorting circuit summary

### Appendix Sections Pushed
- Appendix A: Initialization scale sweep
- Appendix B: Split attn1 (B.1 Heatmaps, B.2 Score Slices for n=2,3,4, B.3 Argmax Bias)
