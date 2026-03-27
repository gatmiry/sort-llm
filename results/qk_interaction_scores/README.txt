How to reproduce: qk_interaction_scores.png
============================================

This plot shows the interaction between key and query projections
coming from the token embeddings (Wte @ Q^T @ K @ Wte^T) for a model
trained with positional embeddings. Each subplot corresponds to a
different token embedding row (10, 30, 50, 70, 90, 110), with a
vertical red dashed line marking the query token's own index.

Quick method (standalone script):
---------------------------------
From the sort-llm root directory, run:

    python results/qk_interaction_scores/generate_plot.py

This will regenerate the plot and save it to:
    results/qk_interaction_scores/qk_interaction_scores.png

Alternative method (notebook):
------------------------------
1. Open the notebook: newtest.ipynb (in the sort-llm root directory).

2. Run Cell 1 (the cell that loads the model). It uses:
   - Model: model_tbyt_3.py (GPT class)
   - Checkpoint: saved_models/2026-03-14_18-40-26_vocab128/march14-withoutlayernorm-block_size:32-batch_size:4096-n_embd:64_head:1_layers:2_vocab_size:128_itr:80000_checkpoint.pt
   - Config: block_size=32, vocab_size=128, without_pos=True

3. Run Cell 5 (the cell that computes the scores matrix). It extracts
   Q, K weight matrices from the first attention layer and computes:
     scores = (Wte @ Q^T) / ||Wte + Wpe|| @ K @ Wte^T / ||Wpe + Wte||

4. Run Cell 7. It produces the 1x6 subplot figure for
   rows = [10, 30, 50, 70, 90, 110] and saves it to:
     results/qk_interaction_scores/qk_interaction_scores.png
