compare_cinclogits.png
======================
Compares two metrics between models trained with and without layer normalization:
  1. Fraction of positions with incorrect attention scores (averaged over all output positions).
  2. Logit correction ratio among incorrect scores: the fraction of incorrect-score positions
     where the model's logit still produces the correct prediction
     (clogit_icscore / (clogit_icscore + iclogit_icscore)).

How to regenerate
-----------------
1. cd sort-llm/

2. Generate the per-location data for both model variants by running
   next_num_statistics.py twice (once with wlnorm='without', once with wlnorm='with'):

     # In next_num_statistics.py, set wlnorm = 'without', then:
     python next_num_statistics.py

     # Change wlnorm = 'with', then:
     python next_num_statistics.py

   This saves statistics_data.npz in plots_withoutlayernorm/ and plots_withlayernorm/.

3. Generate the comparison plot:

     python compare_cinclogits.py

   Output: plots_comparison/compare_cinclogits.png

4. Copy the updated plot here:

     cp plots_comparison/compare_cinclogits.png results/cinclogits/
