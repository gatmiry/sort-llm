#!/usr/bin/env python3
"""
Deep analysis of WHEN L1 attention information matters for L2 key matching.

For each (sorted-output query, unsorted-input key) prediction:
  - Run normal model → top-1 prediction
  - Run with embed-only MLP at keys → top-1 prediction
  - Classify: "both_correct", "attn_helps", "both_wrong", "attn_hurts"

For each case, record features:
  - sorted_position (rank in output)
  - target_value
  - gap_to_prev (target - previous sorted value)
  - gap_to_next (next sorted value - target)
  - min_gap (min of gap_to_prev, gap_to_next)
  - n_within_3, n_within_5, n_within_10 (density around target)
  - l1_self_attn at the correct key position
  - l1_cross_attn_from_nearby_values at the correct key position

Then compare feature distributions between categories.
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
parser.add_argument("--n-trials", type=int, default=800)
ARGS = parser.parse_args()


@torch.no_grad()
def get_components(model, idx):
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
    return att.squeeze(0).squeeze(0)


@torch.no_grad()
def main():
    model = load_model_from_checkpoint(ARGS.ckpt)
    block_size = model.config.block_size
    vocab_n = model.config.vocab_size - 1
    block0 = model.transformer.h[0]

    output_path = ARGS.output or os.path.join(
        os.path.dirname(ARGS.ckpt), "..", "plots", "attn1_importance_analysis.png"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    enable_attention_storage(model)
    n_trials = ARGS.n_trials

    records = []  # list of dicts, one per prediction

    for trial in range(n_trials):
        idx = get_batch(1, block_size, DEVICE, vocab_n=vocab_n)
        embed, attn0_out, mlp0_out = get_components(model, idx)

        sorted_tokens = idx[0, block_size+1:].cpu().numpy()
        unsorted_tokens = idx[0, :block_size].cpu().numpy()
        input_set = set(int(v) for v in unsorted_tokens)

        val_to_pos = {}
        for p in range(block_size):
            val_to_pos[int(unsorted_tokens[p])] = p

        # Normal L2 attention
        residual_normal = embed + attn0_out + mlp0_out
        l2_normal = compute_l2_attention(model, residual_normal)

        # Embed-only MLP L2 attention — only modify key positions (unsorted)
        mlp_embed = block0.mlp(block0.ln_2(embed))
        mod_mlp = mlp0_out.clone()
        mod_mlp[:, :block_size, :] = mlp_embed[:, :block_size, :]
        residual_embed = embed + attn0_out + mod_mlp
        l2_embed = compute_l2_attention(model, residual_embed)

        # Get L1 attention (run forward to populate stored attention)
        model(idx, block_size=block_size)
        l1_attn = model.transformer.h[0].attn.attn.clone().cpu().numpy()  # (T, T)

        for p in range(block_size - 1):
            query_pos = block_size + 1 + p
            target_val = int(sorted_tokens[p + 1])
            if target_val not in val_to_pos:
                continue
            key_pos = val_to_pos[target_val]

            # Top-1 predictions
            normal_top1 = int(np.argmax(l2_normal[query_pos, :block_size].cpu().numpy()))
            embed_top1 = int(np.argmax(l2_embed[query_pos, :block_size].cpu().numpy()))

            normal_correct = (normal_top1 == key_pos)
            embed_correct = (embed_top1 == key_pos)

            if normal_correct and embed_correct:
                category = "both_correct"
            elif normal_correct and not embed_correct:
                category = "attn_helps"
            elif not normal_correct and not embed_correct:
                category = "both_wrong"
            else:
                category = "attn_hurts"

            # Features
            prev_val = int(sorted_tokens[p]) if p >= 0 else -1
            next_val = int(sorted_tokens[p + 2]) if p + 2 < block_size else vocab_n + 1
            gap_prev = target_val - prev_val if prev_val >= 0 else 999
            gap_next = next_val - target_val if next_val <= vocab_n else 999
            min_gap = min(gap_prev, gap_next)

            # Density: how many input numbers within various distances
            n_within = {}
            for dist in [1, 2, 3, 5, 10, 20]:
                count = sum(1 for v in input_set
                            if v != target_val and abs(v - target_val) <= dist)
                n_within[dist] = count

            # L1 attention at the correct key position
            l1_self = l1_attn[key_pos, key_pos]
            l1_total_from_unsorted = l1_attn[key_pos, :block_size].sum()

            # L1 attention from positions containing nearby values (within 5)
            l1_from_nearby = 0.0
            for v in input_set:
                if v != target_val and abs(v - target_val) <= 5:
                    nearby_pos = val_to_pos.get(v)
                    if nearby_pos is not None and nearby_pos < key_pos:
                        l1_from_nearby += l1_attn[key_pos, nearby_pos]

            # What does embed-only model attend to instead?
            embed_attended_val = int(unsorted_tokens[embed_top1])
            embed_error_dist = abs(embed_attended_val - target_val) if not embed_correct else 0

            # Attention weights on correct key
            normal_attn_weight = l2_normal[query_pos, key_pos].item()
            embed_attn_weight = l2_embed[query_pos, key_pos].item()

            # Sorted position (normalized)
            sorted_position = (p + 1) / block_size  # 0 to 1

            records.append({
                "category": category,
                "sorted_pos": p + 1,
                "sorted_pos_norm": sorted_position,
                "target_val": target_val,
                "gap_prev": gap_prev,
                "gap_next": gap_next,
                "min_gap": min_gap,
                **{f"n_within_{d}": n_within[d] for d in [1, 2, 3, 5, 10, 20]},
                "l1_self_attn": l1_self,
                "l1_from_nearby": l1_from_nearby,
                "normal_attn_wt": normal_attn_weight,
                "embed_attn_wt": embed_attn_weight,
                "embed_error_dist": embed_error_dist,
                "embed_attended_val": embed_attended_val,
            })

        if (trial + 1) % 200 == 0:
            print(f"  {trial+1}/{n_trials}")

    # Convert to arrays for easy analysis
    cats = [r["category"] for r in records]
    cat_counts = {c: cats.count(c) for c in ["both_correct", "attn_helps", "both_wrong", "attn_hurts"]}
    total = len(records)

    print(f"\nTotal predictions: {total}")
    for c, n in cat_counts.items():
        print(f"  {c:15s}: {n:6d} ({n/total:.1%})")

    # Split records by category
    by_cat = {c: [r for r in records if r["category"] == c]
              for c in ["both_correct", "attn_helps", "both_wrong", "attn_hurts"]}

    def get_feature(cat, key):
        return np.array([r[key] for r in by_cat[cat]]) if by_cat[cat] else np.array([])

    # ========== PLOTTING ==========
    fig = plt.figure(figsize=(28, 22))
    gs = fig.add_gridspec(4, 4, hspace=0.45, wspace=0.35)

    cat_colors = {
        "both_correct": "#1b7837",
        "attn_helps":   "#b2182b",
        "both_wrong":   "#2166ac",
        "attn_hurts":   "#d6604d",
    }
    cat_nice = {
        "both_correct": "Both correct",
        "attn_helps":   "Attn helps\n(normal ✓, embed-only ✗)",
        "both_wrong":   "Both wrong",
        "attn_hurts":   "Attn hurts\n(normal ✗, embed-only ✓)",
    }

    # Panel (0,0): Category counts
    ax = fig.add_subplot(gs[0, 0])
    cats_order = ["both_correct", "attn_helps", "both_wrong", "attn_hurts"]
    counts = [cat_counts[c] for c in cats_order]
    bars = ax.bar(range(4), counts, color=[cat_colors[c] for c in cats_order],
                  alpha=0.75, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(4))
    ax.set_xticklabels([cat_nice[c] for c in cats_order], fontsize=7)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title("Category distribution", fontsize=11, fontweight="bold")
    for bar, n in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                f"{n}\n({n/total:.1%})", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel (0,1): min_gap distribution for both_correct vs attn_helps
    ax = fig.add_subplot(gs[0, 1])
    for cat in ["both_correct", "attn_helps"]:
        data = get_feature(cat, "min_gap")
        if len(data) == 0:
            continue
        bins = np.arange(0.5, 40.5, 1)
        ax.hist(data, bins=bins, alpha=0.5, color=cat_colors[cat], density=True,
                label=f"{cat_nice[cat].split(chr(10))[0]} (med={np.median(data):.0f})")
    ax.set_xlabel("Min gap (to prev or next sorted value)", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title("Min gap distribution", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 40)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel (0,2): gap_prev distribution
    ax = fig.add_subplot(gs[0, 2])
    for cat in ["both_correct", "attn_helps"]:
        data = get_feature(cat, "gap_prev")
        data = data[data < 100]
        if len(data) == 0:
            continue
        bins = np.arange(0.5, 50.5, 1)
        ax.hist(data, bins=bins, alpha=0.5, color=cat_colors[cat], density=True,
                label=f"{cat_nice[cat].split(chr(10))[0]} (med={np.median(data):.0f})")
    ax.set_xlabel("Gap to previous sorted value", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title("Gap to previous value", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 50)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel (0,3): n_within_3 distribution
    ax = fig.add_subplot(gs[0, 3])
    for cat in ["both_correct", "attn_helps"]:
        data = get_feature(cat, "n_within_3")
        if len(data) == 0:
            continue
        bins = np.arange(-0.5, max(data.max(), 5) + 1.5, 1)
        ax.hist(data, bins=bins, alpha=0.5, color=cat_colors[cat], density=True,
                label=f"{cat_nice[cat].split(chr(10))[0]} (μ={np.mean(data):.2f})")
    ax.set_xlabel("# input numbers within ±3 of target", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title("Local density (±3)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel (1,0): sorted_pos distribution
    ax = fig.add_subplot(gs[1, 0])
    for cat in ["both_correct", "attn_helps"]:
        data = get_feature(cat, "sorted_pos")
        if len(data) == 0:
            continue
        bins = np.arange(0.5, block_size + 0.5, 1)
        ax.hist(data, bins=bins, alpha=0.5, color=cat_colors[cat], density=True,
                label=f"{cat_nice[cat].split(chr(10))[0]} (med={np.median(data):.0f})")
    ax.set_xlabel("Sorted output position", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title("Where in sorted output?", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel (1,1): n_within_1 (immediate neighbors)
    ax = fig.add_subplot(gs[1, 1])
    for cat in ["both_correct", "attn_helps"]:
        data = get_feature(cat, "n_within_1")
        if len(data) == 0:
            continue
        bins = np.arange(-0.5, max(data.max(), 3) + 1.5, 1)
        ax.hist(data, bins=bins, alpha=0.5, color=cat_colors[cat], density=True,
                label=f"{cat_nice[cat].split(chr(10))[0]} (μ={np.mean(data):.2f})")
    ax.set_xlabel("# input numbers within ±1 of target", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title("Immediate neighbors (±1)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel (1,2): embed_error_dist for attn_helps cases
    ax = fig.add_subplot(gs[1, 2])
    data_helps = get_feature("attn_helps", "embed_error_dist")
    if len(data_helps) > 0:
        bins = np.arange(0.5, min(data_helps.max() + 1.5, 50), 1)
        ax.hist(data_helps, bins=bins, alpha=0.7, color=cat_colors["attn_helps"],
                edgecolor="black", linewidth=0.3)
        ax.set_xlabel("Distance: embed-only attended value vs target", fontsize=9)
        ax.set_ylabel("Count", fontsize=10)
        ax.set_title("When attn helps:\nhow far off is embed-only?", fontsize=11, fontweight="bold")
        med = np.median(data_helps)
        ax.axvline(med, color="black", linewidth=2, linestyle="--", label=f"median={med:.0f}")
        ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel (1,3): L1 self-attention at correct key
    ax = fig.add_subplot(gs[1, 3])
    for cat in ["both_correct", "attn_helps"]:
        data = get_feature(cat, "l1_self_attn")
        if len(data) == 0:
            continue
        ax.hist(data, bins=40, alpha=0.5, color=cat_colors[cat], density=True,
                label=f"{cat_nice[cat].split(chr(10))[0]} (μ={np.mean(data):.3f})")
    ax.set_xlabel("L1 self-attention at correct key pos", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title("L1 self-attention fraction", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel (2,0): Attn weight comparison: normal vs embed-only
    ax = fig.add_subplot(gs[2, 0])
    for cat in ["both_correct", "attn_helps"]:
        x = get_feature(cat, "embed_attn_wt")
        y = get_feature(cat, "normal_attn_wt")
        if len(x) == 0:
            continue
        ax.scatter(x, y, alpha=0.05, s=5, color=cat_colors[cat],
                   label=cat_nice[cat].split(chr(10))[0])
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.5, alpha=0.5)
    ax.set_xlabel("Embed-only attn weight on correct key", fontsize=10)
    ax.set_ylabel("Normal attn weight on correct key", fontsize=10)
    ax.set_title("Attention weight comparison", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel (2,1): Conditional failure rate by min_gap
    ax = fig.add_subplot(gs[2, 1])
    gap_bins = [(1, 1), (2, 2), (3, 3), (4, 5), (6, 10), (11, 20), (21, 50)]
    failure_rates = []
    gap_labels_l = []
    for lo, hi in gap_bins:
        n_total = sum(1 for r in records if lo <= r["min_gap"] <= hi)
        n_helps = sum(1 for r in records if lo <= r["min_gap"] <= hi and r["category"] == "attn_helps")
        n_wrong = sum(1 for r in records if lo <= r["min_gap"] <= hi and r["category"] == "both_wrong")
        if n_total > 0:
            failure_rates.append((n_helps / n_total, n_wrong / n_total, n_total))
            gap_labels_l.append(f"{lo}" if lo == hi else f"{lo}-{hi}")
    if failure_rates:
        x = np.arange(len(failure_rates))
        helps_rates = [f[0] for f in failure_rates]
        wrong_rates = [f[1] for f in failure_rates]
        totals = [f[2] for f in failure_rates]
        ax.bar(x - 0.15, helps_rates, width=0.3, color=cat_colors["attn_helps"], alpha=0.75,
               label="Attn helps rate")
        ax.bar(x + 0.15, wrong_rates, width=0.3, color=cat_colors["both_wrong"], alpha=0.75,
               label="Both wrong rate")
        ax.set_xticks(x)
        ax.set_xticklabels(gap_labels_l, fontsize=8)
        for i, (h, w, n) in enumerate(zip(helps_rates, wrong_rates, totals)):
            ax.text(i, max(h, w) + 0.01, f"n={n}", ha="center", fontsize=6.5, color="gray")
        ax.set_xlabel("Min gap to neighboring sorted value", fontsize=10)
        ax.set_ylabel("Failure rate", fontsize=10)
        ax.set_title("Failure rate by min gap\n(when does attn matter?)", fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel (2,2): Conditional failure rate by n_within_5
    ax = fig.add_subplot(gs[2, 2])
    density_bins = list(range(0, 8))
    fail_by_density = []
    density_labels_l = []
    for d in density_bins:
        n_total = sum(1 for r in records if r["n_within_5"] == d)
        n_helps = sum(1 for r in records if r["n_within_5"] == d and r["category"] == "attn_helps")
        n_wrong = sum(1 for r in records if r["n_within_5"] == d and r["category"] == "both_wrong")
        if n_total > 10:
            fail_by_density.append((n_helps / n_total, n_wrong / n_total, n_total))
            density_labels_l.append(str(d))
    if fail_by_density:
        x = np.arange(len(fail_by_density))
        helps_d = [f[0] for f in fail_by_density]
        wrong_d = [f[1] for f in fail_by_density]
        totals_d = [f[2] for f in fail_by_density]
        ax.bar(x - 0.15, helps_d, width=0.3, color=cat_colors["attn_helps"], alpha=0.75,
               label="Attn helps rate")
        ax.bar(x + 0.15, wrong_d, width=0.3, color=cat_colors["both_wrong"], alpha=0.75,
               label="Both wrong rate")
        ax.set_xticks(x)
        ax.set_xticklabels(density_labels_l, fontsize=8)
        for i, (h, w, n) in enumerate(zip(helps_d, wrong_d, totals_d)):
            ax.text(i, max(h, w) + 0.01, f"n={n}", ha="center", fontsize=6.5, color="gray")
        ax.set_xlabel("# input numbers within ±5 of target", fontsize=10)
        ax.set_ylabel("Failure rate", fontsize=10)
        ax.set_title("Failure rate by local density\n(when does attn matter?)", fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel (2,3): Conditional failure rate by sorted position
    ax = fig.add_subplot(gs[2, 3])
    pos_bins = [(1, 4), (5, 8), (9, 12), (13, 16), (17, 20), (21, 24), (25, 28), (29, 31)]
    fail_by_pos = []
    pos_labels_l = []
    for lo, hi in pos_bins:
        n_total = sum(1 for r in records if lo <= r["sorted_pos"] <= hi)
        n_helps = sum(1 for r in records if lo <= r["sorted_pos"] <= hi and r["category"] == "attn_helps")
        if n_total > 0:
            fail_by_pos.append((n_helps / n_total, n_total))
            pos_labels_l.append(f"{lo}-{hi}")
    if fail_by_pos:
        x = np.arange(len(fail_by_pos))
        ax.bar(x, [f[0] for f in fail_by_pos], color=cat_colors["attn_helps"], alpha=0.75)
        ax.set_xticks(x)
        ax.set_xticklabels(pos_labels_l, fontsize=8)
        for i, (rate, n) in enumerate(fail_by_pos):
            ax.text(i, rate + 0.005, f"{rate:.1%}\n(n={n})", ha="center", fontsize=6.5)
        ax.set_xlabel("Sorted output position", fontsize=10)
        ax.set_ylabel("Attn-helps rate", fontsize=10)
        ax.set_title("Attn-helps rate by position", fontsize=11, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel (3,0): Comparison of n_within_5 and n_within_10
    ax = fig.add_subplot(gs[3, 0])
    features_to_compare = ["n_within_1", "n_within_2", "n_within_3", "n_within_5", "n_within_10", "n_within_20"]
    feat_labels = ["±1", "±2", "±3", "±5", "±10", "±20"]
    for ci, cat in enumerate(["both_correct", "attn_helps"]):
        means = [np.mean(get_feature(cat, f)) if len(get_feature(cat, f)) > 0 else 0
                 for f in features_to_compare]
        x = np.arange(len(features_to_compare))
        ax.bar(x + ci * 0.35, means, width=0.35, color=cat_colors[cat], alpha=0.75,
               label=cat_nice[cat].split(chr(10))[0])
    ax.set_xticks(np.arange(len(feat_labels)) + 0.175)
    ax.set_xticklabels(feat_labels, fontsize=8)
    ax.set_xlabel("Neighborhood radius", fontsize=10)
    ax.set_ylabel("Mean # neighbors", fontsize=10)
    ax.set_title("Mean local density by category", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel (3,1): L1 attention from nearby values
    ax = fig.add_subplot(gs[3, 1])
    for cat in ["both_correct", "attn_helps"]:
        data = get_feature(cat, "l1_from_nearby")
        if len(data) == 0:
            continue
        ax.hist(data, bins=40, alpha=0.5, color=cat_colors[cat], density=True,
                label=f"{cat_nice[cat].split(chr(10))[0]} (μ={np.mean(data):.3f})")
    ax.set_xlabel("L1 attn from positions with nearby values (±5)", fontsize=9)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title("L1 cross-attn from\nnearby-value positions", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel (3,2): gap_prev for attn_helps: how close is the confusing alternative?
    ax = fig.add_subplot(gs[3, 2])
    if len(by_cat["attn_helps"]) > 0:
        gap_prev_h = get_feature("attn_helps", "gap_prev")
        gap_next_h = get_feature("attn_helps", "gap_next")
        gap_prev_h = gap_prev_h[gap_prev_h < 100]
        gap_next_h = gap_next_h[gap_next_h < 100]
        bins = np.arange(0.5, 30.5, 1)
        ax.hist(gap_prev_h, bins=bins, alpha=0.5, color="#b2182b", label=f"Gap prev (med={np.median(gap_prev_h):.0f})")
        ax.hist(gap_next_h, bins=bins, alpha=0.5, color="#2166ac", label=f"Gap next (med={np.median(gap_next_h):.0f})")
        ax.set_xlabel("Gap size", fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.set_title("Gap distribution\n(attn-helps cases only)", fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel (3,3): Summary statistics text
    ax = fig.add_subplot(gs[3, 3])
    ax.axis("off")
    lines = ["SUMMARY STATISTICS", "=" * 40, ""]
    for cat in ["both_correct", "attn_helps", "both_wrong"]:
        n = len(by_cat[cat])
        if n == 0:
            continue
        lines.append(f"─── {cat_nice[cat].split(chr(10))[0]} (n={n}) ───")
        for feat, label in [
            ("min_gap", "Min gap"),
            ("gap_prev", "Gap prev"),
            ("n_within_1", "Neighbors ±1"),
            ("n_within_3", "Neighbors ±3"),
            ("n_within_5", "Neighbors ±5"),
            ("l1_self_attn", "L1 self-attn"),
            ("l1_from_nearby", "L1 from nearby"),
        ]:
            vals = get_feature(cat, feat)
            if len(vals) > 0:
                lines.append(f"  {label:16s}: μ={np.mean(vals):6.2f}  med={np.median(vals):5.1f}")
        lines.append("")

    ax.text(0.02, 0.98, "\n".join(lines), transform=ax.transAxes,
            fontsize=7, verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0", alpha=0.8))

    fig.suptitle(
        f"When does L1 attention matter for L2 key matching?\n"
        f"k={block_size}, N={vocab_n}, {os.path.basename(ARGS.ckpt)}  |  {n_trials} trials, {total} predictions",
        fontsize=14, fontweight="bold",
    )
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
