attention_errors — Three-panel figure on attention score errors and logit correction
======================================================================================

Contents:
  icscore_perlocation.png           — Incorrect first-layer attention scores per position (without LN)
  correction_fraction_withlayernorm.png — One/both attention layers wrong per position (with LN)
  compare_cinclogits.png            — Aggregate incorrect scores & correction ratio comparison

How to regenerate
-----------------

1. cd sort-llm/

2. icscore_perlocation.png:
     # In next_num_statistics.py, set wlnorm = 'without'
     python next_num_statistics.py
     # Output: plots/icscore_perlocation.png

3. correction_fraction_withlayernorm.png:
     # In correctionfraction_layernorm.py, set wlnorm = 'with'
     python correctionfraction_layernorm.py
     # Output: plots/correction_fraction_withlayernorm.png

4. compare_cinclogits.png:
     # Requires data from both models. Run next_num_statistics.py once
     # with wlnorm='without' and once with wlnorm='with' (step 2 above
     # plus repeating with 'with'). Then:
     python compare_cinclogits.py
     # Output: plots_comparison/compare_cinclogits.png

5. Copy updated plots here:
     cp plots/icscore_perlocation.png results/attention_errors/
     cp plots/correction_fraction_withlayernorm.png results/attention_errors/
     cp plots_comparison/compare_cinclogits.png results/attention_errors/
