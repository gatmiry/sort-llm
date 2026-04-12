#!/usr/bin/env python3
"""
Deep causal analysis of Layer 2 QK interaction.

PART 1 — Verify MLP dominance (zero-ablation at KEY positions):
  Modify the residual stream only at unsorted-input positions (keys),
  leaving sorted-output positions (queries) untouched.
  Conditions:
    (a) Normal
    (b) Zero L1 MLP at keys
    (c) Zero L1 Attn at keys (causal: MLP is recomputed from embed only)
    (d) Zero L1 Attn at keys (direct only: MLP output is kept from normal)

PART 2 — Decompose MLP input:
  At unsorted positions, modify the MLP input:
    (e) MLP receives LN_2(embed) only (no L1 attn contribution)
    (f) MLP receives LN_2(attn0_out) only (no embed contribution)
  In both cases, embed and attn0_out remain in the residual via their
  direct paths. Only the MLP output changes.

PART 3 — L1 attention diagnostics:
  At unsorted positions, measure L1 self-attention fraction.

Measurement: for each sorted-output position, what fraction of L2
attention goes to the correct unsorted-input key?
"""

import os, sys, math, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "sort-llm", "sortgpt_toolkit"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from model import DEVICE, load_model_from_checkpoint, get_batch
from intervene import enable_attention_storage

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", required=True)
parser.add_argument("--output", default=None)
parser.add_argument("--n-trials", type=int, default=400)
ARGS = parser.parse_args()


@torch.no_grad()
def get_components(model, idx):
    """Manual forward through Block 0, returning each residual component."""
    block0 = model.transformer.h[0]
    B, T = idx.size()
    pos = model.transformer.wpe(model.pos_idx[:T])
    embed = model.transformer.wte(idx) + pos

    h0 = block0.ln_1(embed)
    attn0_out = block0.attn(h0)
    x_after_attn0 = embed + attn0_out

    h0_mlp = block0.ln_2(x_after_attn0)
    mlp0_out = block0.mlp(h0_mlp)

    return embed, attn0_out, mlp0_out


@torch.no_grad()
def compute_l2_attention(model, residual):
    """Compute L2 attention weights from a (possibly modified) residual."""
    block1 = model.transformer.h[1]
    n_embd = model.config.n_embd
    h = block1.ln_1(residual)
    B, T, C = h.size()

    qkv = block1.attn.c_attn(h)
    q, k, _ = qkv.split(n_embd, dim=2)

    n_heads = block1.attn.n_heads
    head_dim = block1.attn.head_dim
    q = q.view(B, T, n_heads, head_dim).transpose(1, 2)
    k = k.view(B, T, n_heads, head_dim).transpose(1, 2)

    scale = 1.0 / math.sqrt(head_dim)
    att = (q @ k.transpose(-2, -1)) * scale

    causal = torch.triu(torch.ones(T, T, device=residual.device, dtype=torch.bool), diagonal=1)
    att = att.masked_fill(causal.unsqueeze(0).unsqueeze(0), float("-inf"))
    att = F.softmax(att, dim=-1)
    return att.squeeze(0).squeeze(0)  # (T, T) — single head, single batch


@torch.no_grad()
def build_residual(model, embed, attn0_out, mlp0_out, block_size, condition):
    """Build a (possibly modified) residual for all positions.

    Only unsorted-input positions (0..block_size-1) are modified.
    Sorted-output positions keep their original residual.
    """
    block0 = model.transformer.h[0]

    if condition == "normal":
        return embed + attn0_out + mlp0_out

    elif condition == "zero_mlp_keys":
        mod_mlp = mlp0_out.clone()
        mod_mlp[:, :block_size, :] = 0
        return embed + attn0_out + mod_mlp

    elif condition == "zero_attn_causal":
        # Zero attn0 at keys; recompute MLP from embed only at those positions
        mod_attn = attn0_out.clone()
        mod_attn[:, :block_size, :] = 0
        x_tmp = embed + mod_attn
        recomputed_mlp = block0.mlp(block0.ln_2(x_tmp))
        mod_mlp = mlp0_out.clone()
        mod_mlp[:, :block_size, :] = recomputed_mlp[:, :block_size, :]
        return embed + mod_attn + mod_mlp

    elif condition == "zero_attn_direct":
        # Zero attn0 in residual at keys, but keep original MLP output
        mod_attn = attn0_out.clone()
        mod_attn[:, :block_size, :] = 0
        return embed + mod_attn + mlp0_out

    elif condition == "mlp_embed_only":
        # MLP at keys receives LN_2(embed) instead of LN_2(embed+attn0)
        recomputed_mlp = block0.mlp(block0.ln_2(embed))
        mod_mlp = mlp0_out.clone()
        mod_mlp[:, :block_size, :] = recomputed_mlp[:, :block_size, :]
        return embed + attn0_out + mod_mlp

    elif condition == "mlp_attn_only":
        # MLP at keys receives LN_2(attn0_out) instead of LN_2(embed+attn0)
        recomputed_mlp = block0.mlp(block0.ln_2(attn0_out))
        mod_mlp = mlp0_out.clone()
        mod_mlp[:, :block_size, :] = recomputed_mlp[:, :block_size, :]
        return embed + attn0_out + mod_mlp

    elif condition == "mlp_only":
        # Only MLP output at keys (zero embed and attn0)
        mod_embed = embed.clone()
        mod_embed[:, :block_size, :] = 0
        mod_attn = attn0_out.clone()
        mod_attn[:, :block_size, :] = 0
        return mod_embed + mod_attn + mlp0_out

    elif condition == "mlp_only_from_embed":
        # Only MLP output (computed from embed only) at keys
        recomputed_mlp = block0.mlp(block0.ln_2(embed))
        mod_embed = embed.clone()
        mod_embed[:, :block_size, :] = 0
        mod_attn = attn0_out.clone()
        mod_attn[:, :block_size, :] = 0
        mod_mlp = mlp0_out.clone()
        mod_mlp[:, :block_size, :] = recomputed_mlp[:, :block_size, :]
        return mod_embed + mod_attn + mod_mlp

    else:
        raise ValueError(condition)


@torch.no_grad()
def main():
    model = load_model_from_checkpoint(ARGS.ckpt)
    block_size = model.config.block_size
    vocab_n = model.config.vocab_size - 1
    n_embd = model.config.n_embd

    output_path = ARGS.output or os.path.join(
        os.path.dirname(ARGS.ckpt), "..", "plots", "qk_deep_decomp.png"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    enable_attention_storage(model)

    n_trials = ARGS.n_trials
    conditions = [
        "normal",
        "zero_mlp_keys",
        "zero_attn_causal",
        "zero_attn_direct",
        "mlp_embed_only",
        "mlp_attn_only",
        "mlp_only",
        "mlp_only_from_embed",
    ]
    cond_labels = {
        "normal":              "Normal",
        "zero_mlp_keys":       "Zero MLP\n(at keys)",
        "zero_attn_causal":    "Zero Attn\n(causal)",
        "zero_attn_direct":    "Zero Attn\n(direct only)",
        "mlp_embed_only":      "MLP from\nembed only",
        "mlp_attn_only":       "MLP from\nattn only",
        "mlp_only":            "MLP only\n(no embed/attn)",
        "mlp_only_from_embed": "MLP only\n(from embed)",
    }

    # Accumulators
    # (n_conditions,) — mean L2 attention on correct key
    attn_on_correct = {c: [] for c in conditions}
    # Per sorted position: (n_conditions, block_size-1)
    n_pred_pos = block_size - 1
    per_pos = {c: np.zeros(n_pred_pos) for c in conditions}
    per_pos_count = np.zeros(n_pred_pos)

    # L1 self-attention at unsorted positions
    l1_self_attn_frac = []
    # L1 attention: fraction going to other unsorted positions
    l1_cross_attn_frac = []

    # Cosine similarity between normal MLP output and modified MLP output at keys
    cos_sim_embed_only = []
    cos_sim_attn_only = []

    # QK score decomposition at correct key (from Part 1 of previous experiment)
    qk_comp_scores = {c: [] for c in ["embed", "attn", "mlp", "full"]}

    # Top-1 accuracy: does the max-attention unsorted position contain the target?
    top1_correct = {c: [] for c in conditions}

    block0 = model.transformer.h[0]
    block1 = model.transformer.h[1]
    W_k = block1.attn.c_attn.weight[n_embd:2*n_embd, :]
    b_k = block1.attn.c_attn.bias[n_embd:2*n_embd]
    W_q = block1.attn.c_attn.weight[:n_embd, :]
    b_q = block1.attn.c_attn.bias[:n_embd]
    ln_L2 = block1.ln_1

    for trial in range(n_trials):
        idx = get_batch(1, block_size, DEVICE, vocab_n=vocab_n)
        embed, attn0_out, mlp0_out = get_components(model, idx)

        sorted_tokens = idx[0, block_size+1:].cpu().numpy()
        unsorted_tokens = idx[0, :block_size].cpu().numpy()

        # Map: token value -> unsorted position
        val_to_pos = {}
        for p in range(block_size):
            val_to_pos[int(unsorted_tokens[p])] = p

        # ---- Part 1 & 2: Ablation experiments ----
        # At position k+1+p (containing sorted[p]), the model predicts sorted[p+1].
        # The "correct key" is the unsorted position containing sorted[p+1].
        for cond in conditions:
            residual = build_residual(model, embed, attn0_out, mlp0_out, block_size, cond)
            l2_attn = compute_l2_attention(model, residual)

            trial_correct_attns = []
            trial_top1 = []
            for p in range(n_pred_pos):
                query_pos = block_size + 1 + p  # position of sorted[p]
                target_val = int(sorted_tokens[p + 1])  # predicts sorted[p+1]
                if target_val not in val_to_pos:
                    continue
                key_pos = val_to_pos[target_val]
                attn_weight = l2_attn[query_pos, key_pos].item()
                trial_correct_attns.append(attn_weight)
                per_pos[cond][p] += attn_weight

                # Top-1: is the max-attention unsorted position the correct one?
                unsorted_attn = l2_attn[query_pos, :block_size].cpu().numpy()
                trial_top1.append(int(np.argmax(unsorted_attn) == key_pos))

            if trial_correct_attns:
                attn_on_correct[cond].append(np.mean(trial_correct_attns))
            if trial_top1:
                top1_correct[cond].append(np.mean(trial_top1))

        per_pos_count += 1

        # ---- Part 3: L1 attention diagnostic at unsorted positions ----
        model(idx, block_size=block_size)
        l1_attn = model.transformer.h[0].attn.attn.clone()  # (T, T)
        self_fracs = []
        cross_fracs = []
        for k_pos in range(block_size):
            self_fracs.append(l1_attn[k_pos, k_pos].item())
            # Attention to other unsorted positions (excluding self)
            other_unsorted = l1_attn[k_pos, :block_size].sum().item() - l1_attn[k_pos, k_pos].item()
            cross_fracs.append(other_unsorted)
        l1_self_attn_frac.append(np.mean(self_fracs))
        l1_cross_attn_frac.append(np.mean(cross_fracs))

        # ---- Part 3b: QK score decomposition double-check ----
        full_residual = embed + attn0_out + mlp0_out
        for p in range(n_pred_pos):
            query_pos = block_size + 1 + p
            target_val = int(sorted_tokens[p + 1])
            if target_val not in val_to_pos:
                continue
            key_pos = val_to_pos[target_val]

            # Query from full residual at sorted output position
            h_q = ln_L2(full_residual[:, query_pos:query_pos+1, :])
            Q = (h_q @ W_q.T + b_q).squeeze()  # (n_embd,)

            # Key components at the correct unsorted position
            h_embed = ln_L2(embed[:, key_pos:key_pos+1, :])
            h_attn = ln_L2(attn0_out[:, key_pos:key_pos+1, :])
            h_mlp = ln_L2(mlp0_out[:, key_pos:key_pos+1, :])
            h_full = ln_L2(full_residual[:, key_pos:key_pos+1, :])

            K_embed = (h_embed @ W_k.T + b_k).squeeze()
            K_attn = (h_attn @ W_k.T + b_k).squeeze()
            K_mlp = (h_mlp @ W_k.T + b_k).squeeze()
            K_full = (h_full @ W_k.T + b_k).squeeze()

            qk_comp_scores["embed"].append((Q @ K_embed).item())
            qk_comp_scores["attn"].append((Q @ K_attn).item())
            qk_comp_scores["mlp"].append((Q @ K_mlp).item())
            qk_comp_scores["full"].append((Q @ K_full).item())

        # ---- Cosine similarity: normal MLP vs modified MLP at keys ----
        mlp_normal = mlp0_out[0, :block_size, :]  # (block_size, C)
        mlp_embed_only = block0.mlp(block0.ln_2(embed))[0, :block_size, :]
        mlp_attn_only = block0.mlp(block0.ln_2(attn0_out))[0, :block_size, :]

        cos_e = F.cosine_similarity(mlp_normal, mlp_embed_only, dim=1).mean().item()
        cos_a = F.cosine_similarity(mlp_normal, mlp_attn_only, dim=1).mean().item()
        cos_sim_embed_only.append(cos_e)
        cos_sim_attn_only.append(cos_a)

        if (trial + 1) % 100 == 0:
            print(f"  {trial+1}/{n_trials}")

    # Normalize per-position
    for cond in conditions:
        per_pos[cond] /= per_pos_count

    # ========== PLOTTING ==========
    fig = plt.figure(figsize=(26, 18))
    gs = fig.add_gridspec(3, 4, hspace=0.45, wspace=0.35)

    cond_colors = {
        "normal":              "#2d2d2d",
        "zero_mlp_keys":       "#b2182b",
        "zero_attn_causal":    "#2166ac",
        "zero_attn_direct":    "#4393c3",
        "mlp_embed_only":      "#1b7837",
        "mlp_attn_only":       "#d6604d",
        "mlp_only":            "#762a83",
        "mlp_only_from_embed": "#e08214",
    }

    # ---- Panel (0,0): Bar chart — mean L2 attn weight on correct key ----
    ax = fig.add_subplot(gs[0, 0])
    means = [np.mean(attn_on_correct[c]) for c in conditions]
    stds = [np.std(attn_on_correct[c]) / np.sqrt(len(attn_on_correct[c])) for c in conditions]
    bars = ax.bar(range(len(conditions)), means, yerr=stds,
                  color=[cond_colors[c] for c in conditions], alpha=0.75,
                  edgecolor="black", linewidth=0.5, capsize=3)
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels([cond_labels[c] for c in conditions], fontsize=6.5)
    ax.set_ylabel("L2 attn on correct key", fontsize=9)
    ax.set_title("PART 1: Mean L2 attn weight\non correct next-value key",
                 fontsize=10, fontweight="bold")
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f"{m:.3f}", ha="center", va="bottom", fontsize=6.5, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ---- Panel (0,1): Bar chart — Top-1 accuracy ----
    ax = fig.add_subplot(gs[0, 1])
    t1_means = [np.mean(top1_correct[c]) for c in conditions]
    t1_stds = [np.std(top1_correct[c]) / np.sqrt(len(top1_correct[c])) for c in conditions]
    bars = ax.bar(range(len(conditions)), t1_means, yerr=t1_stds,
                  color=[cond_colors[c] for c in conditions], alpha=0.75,
                  edgecolor="black", linewidth=0.5, capsize=3)
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels([cond_labels[c] for c in conditions], fontsize=6.5)
    ax.set_ylabel("Top-1 accuracy", fontsize=9)
    ax.set_title("PART 1: Top-1 accuracy\n(argmax L2 attn = correct key?)",
                 fontsize=10, fontweight="bold")
    for bar, m in zip(bars, t1_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{m:.2%}", ha="center", va="bottom", fontsize=6.5, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ---- Panel (0,2): Per sorted-output position profile ----
    ax = fig.add_subplot(gs[0, 2:4])
    for cond in conditions:
        ax.plot(range(n_pred_pos), per_pos[cond], linewidth=1.5,
                color=cond_colors[cond], label=cond_labels[cond].replace("\n", " "))
    ax.set_xlabel("Sorted output position (predicting next value)", fontsize=10)
    ax.set_ylabel("L2 attn on correct key", fontsize=10)
    ax.set_title("Per sorted-output position", fontsize=10, fontweight="bold")
    ax.legend(fontsize=6.5, ncol=2, loc="upper right")
    ax.grid(True, alpha=0.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ---- Panel (1,0): QK score decomposition ----
    ax = fig.add_subplot(gs[1, 0])
    comp_labels = ["Embed+Pos", "L1 Attn", "L1 MLP", "Full"]
    comp_keys = ["embed", "attn", "mlp", "full"]
    comp_colors_dc = ["#1b7837", "#2166ac", "#b2182b", "#2d2d2d"]
    comp_means = [np.mean(qk_comp_scores[k]) for k in comp_keys]
    comp_stds = [np.std(qk_comp_scores[k]) / np.sqrt(len(qk_comp_scores[k])) for k in comp_keys]
    bars = ax.bar(range(4), comp_means, yerr=comp_stds,
                  color=comp_colors_dc, alpha=0.75, edgecolor="black", linewidth=0.5, capsize=3)
    ax.set_xticks(range(4))
    ax.set_xticklabels(comp_labels, fontsize=8)
    ax.set_ylabel("QK score", fontsize=9)
    ax.set_title("QK score decomp (correct key)\nLN applied per-component", fontsize=10, fontweight="bold")
    for bar, m in zip(bars, comp_means):
        va_pos = "bottom" if m >= 0 else "top"
        offset = 2 if m >= 0 else -2
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + offset,
                f"{m:.1f}", ha="center", va=va_pos, fontsize=7, fontweight="bold")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.grid(True, axis="y", alpha=0.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ---- Panel (1,1): Focus — MLP input decomposition ----
    ax = fig.add_subplot(gs[1, 1])
    mlp_conds = ["normal", "mlp_embed_only", "mlp_attn_only"]
    mlp_means = [np.mean(attn_on_correct[c]) for c in mlp_conds]
    mlp_stds = [np.std(attn_on_correct[c]) / np.sqrt(len(attn_on_correct[c])) for c in mlp_conds]
    bars = ax.bar(range(3), mlp_means, yerr=mlp_stds,
                  color=[cond_colors[c] for c in mlp_conds], alpha=0.75,
                  edgecolor="black", linewidth=0.5, capsize=3)
    ax.set_xticks(range(3))
    ax.set_xticklabels([cond_labels[c] for c in mlp_conds], fontsize=8)
    ax.set_ylabel("L2 attn on correct key", fontsize=9)
    ax.set_title("PART 2: What drives MLP?\n(MLP input modification at keys)",
                 fontsize=10, fontweight="bold")
    for bar, m in zip(bars, mlp_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f"{m:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ---- Panel (1,2): Cosine similarity ----
    ax = fig.add_subplot(gs[1, 2])
    cos_data = [cos_sim_embed_only, cos_sim_attn_only]
    cos_labels_l = ["Normal vs\nembed-only MLP", "Normal vs\nattn-only MLP"]
    cos_cols = ["#1b7837", "#d6604d"]
    bp = ax.boxplot(cos_data, patch_artist=True, widths=0.5)
    for patch, color in zip(bp["boxes"], cos_cols):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_xticklabels(cos_labels_l, fontsize=8)
    ax.set_ylabel("Cosine similarity", fontsize=9)
    ax.set_title("MLP output similarity\n(normal vs modified input)", fontsize=10, fontweight="bold")
    for i, d in enumerate(cos_data):
        ax.text(i+1, np.median(d) + 0.02, f"med={np.median(d):.3f}",
                ha="center", fontsize=8, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ---- Panel (1,3): L1 attention diagnostics ----
    ax = fig.add_subplot(gs[1, 3])
    ax.hist(l1_self_attn_frac, bins=40, color="#2166ac", alpha=0.5, edgecolor="none",
            label=f"Self-attn (μ={np.mean(l1_self_attn_frac):.3f})", density=True)
    ax.hist(l1_cross_attn_frac, bins=40, color="#b2182b", alpha=0.5, edgecolor="none",
            label=f"Cross-attn (μ={np.mean(l1_cross_attn_frac):.3f})", density=True)
    ax.set_xlabel("Attention fraction", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title("PART 3: L1 attn at unsorted pos\nSelf vs cross (other unsorted)",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ---- Row 2 ----
    # Panel (2,0:2): Distribution
    ax = fig.add_subplot(gs[2, 0:2])
    hist_conds = ["normal", "zero_mlp_keys", "mlp_embed_only", "mlp_attn_only", "mlp_only", "mlp_only_from_embed"]
    for cond in hist_conds:
        data = attn_on_correct[cond]
        ax.hist(data, bins=40, alpha=0.4, color=cond_colors[cond],
                label=f"{cond_labels[cond].replace(chr(10), ' ')} (μ={np.mean(data):.3f})",
                density=True, edgecolor="none")
    ax.set_xlabel("L2 attention on correct key (per trial mean)", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title("Distribution of L2 attention accuracy", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel (2,2:4): Per-position MLP decomp
    ax = fig.add_subplot(gs[2, 2:4])
    for cond in ["normal", "mlp_embed_only", "mlp_only", "mlp_only_from_embed", "zero_mlp_keys"]:
        ax.plot(range(n_pred_pos), per_pos[cond], linewidth=2.0,
                color=cond_colors[cond], label=cond_labels[cond].replace("\n", " "))
    ax.set_xlabel("Sorted output position", fontsize=10)
    ax.set_ylabel("L2 attn on correct key", fontsize=10)
    ax.set_title("Per-position: MLP input decomposition + zero-MLP baseline",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle(
        f"Deep causal analysis of L2 QK interaction\n"
        f"k={block_size}, N={vocab_n}, {os.path.basename(ARGS.ckpt)}  |  {n_trials} trials",
        fontsize=14, fontweight="bold",
    )
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")

    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    print("\n--- PART 1: Ablation at key positions ---")
    print(f"  {'Condition':25s} {'Attn weight':>12s} {'Top-1 acc':>12s}")
    for cond in conditions:
        m_a = np.mean(attn_on_correct[cond])
        m_t = np.mean(top1_correct[cond])
        print(f"  {cond_labels[cond].replace(chr(10), ' '):25s}: {m_a:.4f}       {m_t:.2%}")

    print("\n--- QK score decomposition at correct key (LN per-component) ---")
    for k, label in zip(comp_keys, comp_labels):
        m = np.mean(qk_comp_scores[k])
        print(f"  K = {label:12s}: {m:.1f}")

    print(f"\n--- L1 attention at unsorted positions ---")
    print(f"  Mean self-attn fraction:  {np.mean(l1_self_attn_frac):.4f}")
    print(f"  Mean cross-attn fraction: {np.mean(l1_cross_attn_frac):.4f}")

    print(f"\n--- MLP output cosine similarity ---")
    print(f"  Normal vs embed-only:  median={np.median(cos_sim_embed_only):.4f}")
    print(f"  Normal vs attn-only:   median={np.median(cos_sim_attn_only):.4f}")

    print("\n--- KEY INSIGHTS ---")
    normal_acc = np.mean(attn_on_correct["normal"])
    normal_t1 = np.mean(top1_correct["normal"])
    zero_mlp_acc = np.mean(attn_on_correct["zero_mlp_keys"])
    zero_mlp_t1 = np.mean(top1_correct["zero_mlp_keys"])
    zero_attn_acc = np.mean(attn_on_correct["zero_attn_causal"])
    zero_attn_dir_acc = np.mean(attn_on_correct["zero_attn_direct"])
    mlp_embed_acc = np.mean(attn_on_correct["mlp_embed_only"])
    mlp_attn_acc = np.mean(attn_on_correct["mlp_attn_only"])

    print(f"  Normal: attn={normal_acc:.4f}, top1={normal_t1:.2%}")
    print(f"  Zero MLP at keys: attn={zero_mlp_acc:.4f} "
          f"(Δ={zero_mlp_acc - normal_acc:+.4f}), top1={zero_mlp_t1:.2%}")
    print(f"  Zero Attn (causal): attn={zero_attn_acc:.4f} "
          f"(Δ={zero_attn_acc - normal_acc:+.4f})")
    print(f"  Zero Attn (direct): attn={zero_attn_dir_acc:.4f} "
          f"(Δ={zero_attn_dir_acc - normal_acc:+.4f})")
    print(f"  MLP from embed only: attn={mlp_embed_acc:.4f} "
          f"(Δ={mlp_embed_acc - normal_acc:+.4f})")
    print(f"  MLP from attn only: attn={mlp_attn_acc:.4f} "
          f"(Δ={mlp_attn_acc - normal_acc:+.4f})")


if __name__ == "__main__":
    main()
