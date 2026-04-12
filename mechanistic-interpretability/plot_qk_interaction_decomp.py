#!/usr/bin/env python3
"""
Decompose Layer 2 QK interaction by residual-stream components.

At each sorted-output query position p, we define:
  Q = W_Q_L2 @ LN_L2( v_p )
where v_p is either the L1-attention output or L1-MLP output at position p.

At each unsorted-input key position k, we decompose the residual stream
into three components:
  (1) embed  = token_embedding + positional_embedding
  (2) attn0  = Layer-1 attention output (c_proj included)
  (3) mlp0   = Layer-1 MLP output
and compute:
  K_component = W_K_L2 @ LN_L2( component_k )

The interaction score is  Q_p · K_component_k.

We average over many random inputs and compare which key component
has the largest interaction with the query.
"""

import os, sys, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "sort-llm", "sortgpt_toolkit"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from model import DEVICE, load_model_from_checkpoint, get_batch

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", required=True)
parser.add_argument("--output", default=None)
parser.add_argument("--n-trials", type=int, default=300)
ARGS = parser.parse_args()


@torch.no_grad()
def manual_forward_decomposed(model, idx):
    """Run the model manually through Block 0, returning each residual component."""
    block0 = model.transformer.h[0]

    B, T = idx.size()
    pos = model.transformer.wpe(model.pos_idx[:T])
    embed = model.transformer.wte(idx) + pos  # (B, T, C)

    h0 = block0.ln_1(embed)
    attn0_out = block0.attn(h0)                  # (B, T, C)
    x_after_attn0 = embed + attn0_out

    h0_mlp = block0.ln_2(x_after_attn0)
    mlp0_out = block0.mlp(h0_mlp)                # (B, T, C)

    return embed, attn0_out, mlp0_out


@torch.no_grad()
def main():
    model = load_model_from_checkpoint(ARGS.ckpt)
    n_embd = model.config.n_embd
    vocab_n = model.config.vocab_size - 1
    block_size = model.config.block_size
    block1 = model.transformer.h[1]
    ln_L2 = block1.ln_1

    output_path = ARGS.output or os.path.join(
        os.path.dirname(ARGS.ckpt), "..", "plots", "qk_interaction_decomp.png"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    W_q = block1.attn.c_attn.weight[:n_embd, :]
    W_k = block1.attn.c_attn.weight[n_embd:2*n_embd, :]
    b_q = block1.attn.c_attn.bias[:n_embd]
    b_k = block1.attn.c_attn.bias[n_embd:2*n_embd]

    n_trials = ARGS.n_trials
    n_query_types = 2   # 0: L1 attn out, 1: L1 MLP out
    n_key_comps = 3     # 0: embed, 1: L1 attn out, 2: L1 MLP out

    # Accumulators: (n_query_types, n_key_comps)
    # "all keys" = average over ALL unsorted key positions
    # "correct key" = only the key position containing the target value
    all_keys_abs  = np.zeros((n_query_types, n_key_comps))
    all_keys_sign = np.zeros((n_query_types, n_key_comps))
    correct_key_abs  = np.zeros((n_query_types, n_key_comps))
    correct_key_sign = np.zeros((n_query_types, n_key_comps))

    # Per sorted-position profiles (average over keys)
    per_pos_abs = np.zeros((n_query_types, n_key_comps, block_size))

    # Distributions for box plot: collect per-trial averages
    dist_all = np.zeros((n_query_types, n_key_comps, n_trials))
    dist_correct = np.zeros((n_query_types, n_key_comps, n_trials))

    # Also: ratio of |component score| / |full score| at correct key
    ratio_correct = np.zeros((n_query_types, n_key_comps, n_trials))

    for trial in range(n_trials):
        idx = get_batch(1, block_size, DEVICE, vocab_n=vocab_n)
        embed, attn0_out, mlp0_out = manual_forward_decomposed(model, idx)

        unsorted_tokens = idx[0, :block_size]
        sorted_tokens = idx[0, block_size+1:]
        sorted_vals = sorted_tokens.cpu().numpy()
        unsorted_vals = unsorted_tokens.cpu().numpy()

        # Build a map: token value -> unsorted input position
        val_to_pos = {}
        for pos_idx_k in range(block_size):
            val_to_pos[int(unsorted_vals[pos_idx_k])] = pos_idx_k

        # Key components at unsorted positions: (block_size, C)
        key_embed = ln_L2(embed[0, :block_size, :])
        key_attn  = ln_L2(attn0_out[0, :block_size, :])
        key_mlp   = ln_L2(mlp0_out[0, :block_size, :])
        key_parts = [key_embed, key_attn, key_mlp]

        K_parts = [(kp @ W_k.T + b_k) for kp in key_parts]  # each (block_size, C)

        # Full key (for ratio computation)
        full_residual = embed[0, :block_size, :] + attn0_out[0, :block_size, :] + mlp0_out[0, :block_size, :]
        K_full = ln_L2(full_residual) @ W_k.T + b_k  # (block_size, C)

        # Query components at sorted output positions: (block_size, C)
        sorted_slice = slice(block_size + 1, 2 * block_size + 1)
        q_attn = ln_L2(attn0_out[0, sorted_slice, :])
        q_mlp  = ln_L2(mlp0_out[0, sorted_slice, :])
        Q_parts = [q_attn @ W_q.T + b_q, q_mlp @ W_q.T + b_q]

        for qi, Q in enumerate(Q_parts):
            # Q: (block_size, C), K: (block_size, C)
            # Full score matrix: (block_size_q, block_size_k) = Q @ K^T
            for ki, K_comp in enumerate(K_parts):
                scores = (Q @ K_comp.T)  # (bs_q, bs_k)
                scores_np = scores.cpu().numpy()

                # Average over all keys
                mean_abs = np.abs(scores_np).mean()
                mean_sign = scores_np.mean()
                all_keys_abs[qi, ki] += mean_abs
                all_keys_sign[qi, ki] += mean_sign
                dist_all[qi, ki, trial] = mean_abs

                # Per query position: mean over keys
                per_pos_abs[qi, ki, :] += np.abs(scores_np).mean(axis=1)

                # Correct key only: sorted_pos p predicts sorted_vals[p]
                correct_scores = []
                for p in range(block_size):
                    target_val = int(sorted_vals[p])
                    if target_val in val_to_pos:
                        k_pos = val_to_pos[target_val]
                        correct_scores.append(scores_np[p, k_pos])
                if correct_scores:
                    ca = np.abs(correct_scores).mean()
                    cs = np.mean(correct_scores)
                    correct_key_abs[qi, ki] += ca
                    correct_key_sign[qi, ki] += cs
                    dist_correct[qi, ki, trial] = ca

            # Full score at correct key (for ratio)
            Q_full_score = (Q @ K_full.T)  # (bs_q, bs_k)
            Q_full_np = Q_full_score.cpu().numpy()
            for ki, K_comp in enumerate(K_parts):
                scores_np = (Q @ K_comp.T).cpu().numpy()
                ratios = []
                for p in range(block_size):
                    target_val = int(sorted_vals[p])
                    if target_val in val_to_pos:
                        k_pos = val_to_pos[target_val]
                        full_s = Q_full_np[p, k_pos]
                        comp_s = scores_np[p, k_pos]
                        if abs(full_s) > 1e-6:
                            ratios.append(comp_s / full_s)
                if ratios:
                    ratio_correct[qi, ki, trial] = np.mean(ratios)

        if (trial + 1) % 100 == 0:
            print(f"  {trial+1}/{n_trials}")

    # Normalize accumulators
    all_keys_abs /= n_trials
    all_keys_sign /= n_trials
    correct_key_abs /= n_trials
    correct_key_sign /= n_trials
    per_pos_abs /= n_trials

    q_labels = [r"$Q$ = L1 Attn output", r"$Q$ = L1 MLP output"]
    k_labels = ["Embed+Pos", "L1 Attn out", "L1 MLP out"]
    k_colors = ["#1b7837", "#2166ac", "#b2182b"]

    fig, axes = plt.subplots(2, 4, figsize=(22, 10))

    for qi in range(2):
        # Col 0: bar chart — mean |score| over all unsorted keys
        ax = axes[qi][0]
        vals = all_keys_abs[qi]
        bars = ax.bar(range(3), vals, color=k_colors, alpha=0.75, edgecolor="black", linewidth=0.5)
        ax.set_xticks(range(3))
        ax.set_xticklabels(k_labels, fontsize=8)
        ax.set_ylabel("Mean |score|", fontsize=10)
        ax.set_title(f"{q_labels[qi]}\nAll unsorted keys", fontsize=10, fontweight="bold")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*vals.max(),
                    f"{v:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.15)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Col 1: bar chart — mean |score| at correct key position
        ax = axes[qi][1]
        vals = correct_key_abs[qi]
        bars = ax.bar(range(3), vals, color=k_colors, alpha=0.75, edgecolor="black", linewidth=0.5)
        ax.set_xticks(range(3))
        ax.set_xticklabels(k_labels, fontsize=8)
        ax.set_ylabel("Mean |score|", fontsize=10)
        ax.set_title(f"{q_labels[qi]}\nCorrect key only", fontsize=10, fontweight="bold")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*vals.max(),
                    f"{v:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.15)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Col 2: mean ratio (component / full) at correct key
        ax = axes[qi][2]
        mean_ratios = ratio_correct[qi].mean(axis=1)
        bars = ax.bar(range(3), mean_ratios, color=k_colors, alpha=0.75, edgecolor="black", linewidth=0.5)
        ax.set_xticks(range(3))
        ax.set_xticklabels(k_labels, fontsize=8)
        ax.set_ylabel("score_comp / score_full", fontsize=10)
        ax.set_title(f"{q_labels[qi]}\nFraction of full score (correct key)", fontsize=10, fontweight="bold")
        ax.axhline(y=0, color="black", linewidth=0.5)
        for bar, v in zip(bars, mean_ratios):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.15)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Col 3: per sorted-output position profile (all unsorted keys)
        ax = axes[qi][3]
        for ki in range(3):
            ax.plot(range(block_size), per_pos_abs[qi, ki, :],
                    linewidth=1.5, color=k_colors[ki], label=k_labels[ki])
        ax.set_xlabel("Sorted output position", fontsize=10)
        ax.set_ylabel("Mean |score|", fontsize=10)
        ax.set_title(f"{q_labels[qi]}\nPer sorted pos (all keys)", fontsize=10, fontweight="bold")
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(True, alpha=0.15)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        f"Layer 2 QK interaction decomposition by residual-stream component\n"
        f"k={block_size}, N={vocab_n}, {os.path.basename(ARGS.ckpt)}  |  "
        f"{n_trials} trials  |  Key: LN applied to each component individually",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")

    # Print summary
    print("\n=== Summary ===")
    for qi, ql in enumerate(q_labels):
        print(f"\n{ql}:")
        print(f"  All unsorted keys (mean |score|):")
        for ki, kl in enumerate(k_labels):
            print(f"    {kl:15s}: {all_keys_abs[qi, ki]:.3f}")
        print(f"  Correct key (mean |score|):")
        for ki, kl in enumerate(k_labels):
            print(f"    {kl:15s}: {correct_key_abs[qi, ki]:.3f}")
        print(f"  Correct key (component/full ratio):")
        for ki, kl in enumerate(k_labels):
            print(f"    {kl:15s}: {ratio_correct[qi, ki].mean():.3f}")


if __name__ == "__main__":
    main()
