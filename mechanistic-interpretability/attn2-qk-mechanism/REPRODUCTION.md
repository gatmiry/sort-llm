# Reproduction Guide: Attn2 QK Mechanism Figures

All figures use checkpoint `sort-llm/new-grid/k32_N512/checkpoints/std0p01_iseed1__ckpt100000.pt` (k=32, N=512, seed 1, 100k iterations).

The generation script is `generate_all_plots.py` in this folder. Run from the repository root:

```bash
python3 sort-llm/mechanistic-interpretability/attn2-qk-mechanism/generate_all_plots.py
```

## Figure Descriptions

### 1. `attn_error_rates.png`
**What:** Bar chart comparing attn1 vs attn2 top-1 error rates.  
**Method:** Run 3,000 random sequences through the model, recording stored attention weights. For each sorted output position, check whether the highest-probability key (among unsorted input positions) corresponds to the correct next sorted token. Report the fraction of mismatches per layer.  
**Key result:** Attn1 error ≈ 13.8%, attn2 error ≈ 0.5%.

### 2. `qk_heatmap_asymmetry.png`
**What:** Two heatmaps showing attn2 QK scores under varying attn1 context.  
**Method:** Exploit the fact that attn2 scores depend only on MLP1 output. Construct the synthetic score function `s(z, x, t, y) = q(z, x) · k(t, y)` where:
- `q(z, x) = W_Q^L2 · LN_L2(MLP1(LN_0(e_z + V^L1_x)))` (query vector for query token z with attn1 attending to x)
- `k(t, y) = W_K^L2 · LN_L2(MLP1(LN_0(e_t + V^L1_y)))` (key vector for key token t with attn1 attending to y)

**Left panel:** Fix z=250, x=251 (query-side attn1 target), sweep y (key-side attn1 target) on y-axis and t (key token) on x-axis. Shows score is invariant to y.  
**Right panel:** Fix z=250, y=250 (key-side attn1 target), sweep x (query-side attn1 target) on y-axis and t on x-axis. Shows score depends strongly on x.

### 3. `qk_score_slices_band.png`
**What:** 1D slices of the right heatmap panel from Fig 2, showing the windowed monotonicity.  
**Method:** From the heatmap `s(z=250, x, t, y=250)`, extract horizontal slices at x ∈ {260, 270, 280, 290, 300} and plot score vs key token t. Annotate the argmax of each curve.  
**Key result:** Scores are elevated in a band [z, x] with approximate monotonicity; argmax lies slightly below x (e.g., x=260 → argmax t=255).

### 4. `argmax_saturation.png`
**What:** Argmax key token as a function of query-side attn1 target x, for three query tokens.  
**Method:** For z ∈ {100, 250, 400} and y=250, compute `argmax_t s(z, x, t, y)` for x ranging from z to z+260 (step 2). Plot argmax vs x alongside the t=x diagonal.  
**Key result:** The argmax initially tracks x but saturates — attn2 ignores implausibly large attn1 targets. For z=250, saturation occurs around t ≈ 300.

### 5. `l1_vs_l2_qk_smoothness.png`
**What:** Overlay of attn1 and attn2 QK score profiles.  
**Method:** Compute attn1 QK scores for query token 250 vs all key tokens using `q_L1(250) · k_L1(t)` for all t (raw embeddings through LN and L1 QK weights). Overlay attn2 QK scores from the heatmap at x ∈ {260, 280, 300}. Use dual y-axes since scales differ.  
**Key result:** Attn1 scores are noisy across the full vocabulary; attn2 scores are smoother within their respective windows.

### 6. `attn_spread_comparison.png`
**What:** Empirical attention spread: candidate set size and numerical distance.  
**Method:** Run 1,000 random sequences, storing attention weights. For each sorted output position and each layer, identify unsorted keys with attention weight > 0.04. Compute (a) the average |query_value − key_value| and (b) the count of such keys. Average over trials for each position.  
**Key result:** Attn2 attends to fewer keys (1.2–2.2 vs 1.5–3) with smaller numerical distance (0.3–3 vs 2–14).
