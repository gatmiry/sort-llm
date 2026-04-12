#!/usr/bin/env python3
"""
Same ablation study as plot_qk_deep_decomp.py, but modifying the
QUERY side (sorted output positions) instead of the KEY side (unsorted input).

Keys (unsorted positions) are always kept at their normal values.
Only the sorted output positions have their residual modified.
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
def build_residual_query_side(model, embed, attn0_out, mlp0_out, block_size, condition):
    """Modify residual only at sorted-output positions (queries).
    Unsorted-input positions (keys) are always kept normal."""
    block0 = model.transformer.h[0]
    sorted_slice = slice(block_size, None)  # includes SEP + sorted output

    if condition == "normal":
        return embed + attn0_out + mlp0_out

    elif condition == "zero_mlp_queries":
        mod_mlp = mlp0_out.clone()
        mod_mlp[:, sorted_slice, :] = 0
        return embed + attn0_out + mod_mlp

    elif condition == "zero_attn_causal_q":
        mod_attn = attn0_out.clone()
        mod_attn[:, sorted_slice, :] = 0
        x_tmp = embed + mod_attn
        recomputed_mlp = block0.mlp(block0.ln_2(x_tmp))
        mod_mlp = mlp0_out.clone()
        mod_mlp[:, sorted_slice, :] = recomputed_mlp[:, sorted_slice, :]
        return embed + mod_attn + mod_mlp

    elif condition == "zero_attn_direct_q":
        mod_attn = attn0_out.clone()
        mod_attn[:, sorted_slice, :] = 0
        return embed + mod_attn + mlp0_out

    elif condition == "mlp_embed_only_q":
        recomputed_mlp = block0.mlp(block0.ln_2(embed))
        mod_mlp = mlp0_out.clone()
        mod_mlp[:, sorted_slice, :] = recomputed_mlp[:, sorted_slice, :]
        return embed + attn0_out + mod_mlp

    elif condition == "mlp_attn_only_q":
        recomputed_mlp = block0.mlp(block0.ln_2(attn0_out))
        mod_mlp = mlp0_out.clone()
        mod_mlp[:, sorted_slice, :] = recomputed_mlp[:, sorted_slice, :]
        return embed + attn0_out + mod_mlp

    elif condition == "mlp_only_q":
        mod_embed = embed.clone()
        mod_embed[:, sorted_slice, :] = 0
        mod_attn = attn0_out.clone()
        mod_attn[:, sorted_slice, :] = 0
        return mod_embed + mod_attn + mlp0_out

    else:
        raise ValueError(condition)


@torch.no_grad()
def main():
    model = load_model_from_checkpoint(ARGS.ckpt)
    block_size = model.config.block_size
    vocab_n = model.config.vocab_size - 1
    block0 = model.transformer.h[0]

    output_path = ARGS.output or os.path.join(
        os.path.dirname(ARGS.ckpt), "..", "plots", "qk_query_side_decomp.png"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    enable_attention_storage(model)
    n_trials = ARGS.n_trials
    n_pred_pos = block_size - 1

    conditions = [
        "normal",
        "zero_mlp_queries",
        "zero_attn_causal_q",
        "zero_attn_direct_q",
        "mlp_embed_only_q",
        "mlp_attn_only_q",
        "mlp_only_q",
    ]
    cond_labels = {
        "normal":              "Normal",
        "zero_mlp_queries":    "Zero MLP\n(queries)",
        "zero_attn_causal_q":  "Zero Attn\n(causal, Q)",
        "zero_attn_direct_q":  "Zero Attn\n(direct, Q)",
        "mlp_embed_only_q":    "MLP from\nembed (Q)",
        "mlp_attn_only_q":     "MLP from\nattn (Q)",
        "mlp_only_q":          "MLP only\n(Q side)",
    }
    cond_colors = {
        "normal":              "#2d2d2d",
        "zero_mlp_queries":    "#b2182b",
        "zero_attn_causal_q":  "#2166ac",
        "zero_attn_direct_q":  "#4393c3",
        "mlp_embed_only_q":    "#1b7837",
        "mlp_attn_only_q":     "#d6604d",
        "mlp_only_q":          "#762a83",
    }

    attn_on_correct = {c: [] for c in conditions}
    top1_correct = {c: [] for c in conditions}
    per_pos = {c: np.zeros(n_pred_pos) for c in conditions}
    per_pos_count = np.zeros(n_pred_pos)

    # Also: detailed per-prediction records for embed-only-Q analysis
    records = []

    for trial in range(n_trials):
        idx = get_batch(1, block_size, DEVICE, vocab_n=vocab_n)
        embed, attn0_out, mlp0_out = get_components(model, idx)

        sorted_tokens = idx[0, block_size+1:].cpu().numpy()
        unsorted_tokens = idx[0, :block_size].cpu().numpy()
        input_set = set(int(v) for v in unsorted_tokens)

        val_to_pos = {}
        for p in range(block_size):
            val_to_pos[int(unsorted_tokens[p])] = p

        # Compute L2 attention for each condition
        l2_cache = {}
        for cond in conditions:
            residual = build_residual_query_side(model, embed, attn0_out, mlp0_out, block_size, cond)
            l2_cache[cond] = compute_l2_attention(model, residual)

        for cond in conditions:
            l2 = l2_cache[cond]
            trial_attn = []
            trial_top1 = []
            for p in range(n_pred_pos):
                query_pos = block_size + 1 + p
                target_val = int(sorted_tokens[p + 1])
                if target_val not in val_to_pos:
                    continue
                key_pos = val_to_pos[target_val]
                attn_wt = l2[query_pos, key_pos].item()
                trial_attn.append(attn_wt)
                per_pos[cond][p] += attn_wt

                unsorted_attn = l2[query_pos, :block_size].cpu().numpy()
                trial_top1.append(int(np.argmax(unsorted_attn) == key_pos))

            if trial_attn:
                attn_on_correct[cond].append(np.mean(trial_attn))
            if trial_top1:
                top1_correct[cond].append(np.mean(trial_top1))

        # Detailed records for embed-only-Q vs normal
        l2_normal = l2_cache["normal"]
        l2_embed_q = l2_cache["mlp_embed_only_q"]
        for p in range(n_pred_pos):
            query_pos = block_size + 1 + p
            target_val = int(sorted_tokens[p + 1])
            if target_val not in val_to_pos:
                continue
            key_pos = val_to_pos[target_val]

            normal_top1 = int(np.argmax(l2_normal[query_pos, :block_size].cpu().numpy()))
            embed_top1 = int(np.argmax(l2_embed_q[query_pos, :block_size].cpu().numpy()))

            normal_ok = (normal_top1 == key_pos)
            embed_ok = (embed_top1 == key_pos)

            if normal_ok and embed_ok:
                cat = "both_correct"
            elif normal_ok and not embed_ok:
                cat = "attn_helps"
            elif not normal_ok and not embed_ok:
                cat = "both_wrong"
            else:
                cat = "attn_hurts"

            prev_val = int(sorted_tokens[p])
            next_val = int(sorted_tokens[p + 2]) if p + 2 < block_size else vocab_n + 1
            gap_prev = target_val - prev_val
            gap_next = next_val - target_val if next_val <= vocab_n else 999
            min_gap = min(gap_prev, gap_next)
            n_within_5 = sum(1 for v in input_set if v != target_val and abs(v - target_val) <= 5)

            records.append({
                "category": cat,
                "sorted_pos": p + 1,
                "min_gap": min_gap,
                "gap_prev": gap_prev,
                "n_within_5": n_within_5,
            })

        per_pos_count += 1

        if (trial + 1) % 200 == 0:
            print(f"  {trial+1}/{n_trials}")

    for cond in conditions:
        per_pos[cond] /= per_pos_count

    # Category counts
    cats = [r["category"] for r in records]
    cat_counts = {c: cats.count(c) for c in ["both_correct", "attn_helps", "both_wrong", "attn_hurts"]}
    total = len(records)

    # ========== PLOTTING ==========
    fig, axes = plt.subplots(2, 4, figsize=(26, 11))

    # (0,0): Bar chart — attn weight
    ax = axes[0][0]
    means = [np.mean(attn_on_correct[c]) for c in conditions]
    stds = [np.std(attn_on_correct[c]) / np.sqrt(len(attn_on_correct[c])) for c in conditions]
    bars = ax.bar(range(len(conditions)), means, yerr=stds,
                  color=[cond_colors[c] for c in conditions], alpha=0.75,
                  edgecolor="black", linewidth=0.5, capsize=3)
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels([cond_labels[c] for c in conditions], fontsize=6.5)
    ax.set_ylabel("L2 attn on correct key", fontsize=9)
    ax.set_title("Query-side ablation\nMean L2 attn weight", fontsize=10, fontweight="bold")
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f"{m:.3f}", ha="center", va="bottom", fontsize=6.5, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # (0,1): Bar chart — top-1 accuracy
    ax = axes[0][1]
    t1_means = [np.mean(top1_correct[c]) for c in conditions]
    t1_stds = [np.std(top1_correct[c]) / np.sqrt(len(top1_correct[c])) for c in conditions]
    bars = ax.bar(range(len(conditions)), t1_means, yerr=t1_stds,
                  color=[cond_colors[c] for c in conditions], alpha=0.75,
                  edgecolor="black", linewidth=0.5, capsize=3)
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels([cond_labels[c] for c in conditions], fontsize=6.5)
    ax.set_ylabel("Top-1 accuracy", fontsize=9)
    ax.set_title("Query-side ablation\nTop-1 accuracy", fontsize=10, fontweight="bold")
    for bar, m in zip(bars, t1_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{m:.1%}", ha="center", va="bottom", fontsize=6.5, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # (0,2-3): Per-position profiles
    ax = axes[0][2]
    focus = ["normal", "zero_mlp_queries", "zero_attn_causal_q", "mlp_embed_only_q", "mlp_attn_only_q"]
    for cond in focus:
        ax.plot(range(n_pred_pos), per_pos[cond], linewidth=1.5,
                color=cond_colors[cond], label=cond_labels[cond].replace("\n", " "))
    ax.set_xlabel("Sorted output position", fontsize=9)
    ax.set_ylabel("L2 attn on correct key", fontsize=9)
    ax.set_title("Per-position (query-side ablation)", fontsize=10, fontweight="bold")
    ax.legend(fontsize=6, loc="upper right")
    ax.grid(True, alpha=0.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # (0,3): Category distribution (embed-only-Q)
    ax = axes[0][3]
    cats_order = ["both_correct", "attn_helps", "both_wrong", "attn_hurts"]
    cat_colors_d = {"both_correct": "#1b7837", "attn_helps": "#b2182b",
                    "both_wrong": "#2166ac", "attn_hurts": "#d6604d"}
    counts = [cat_counts.get(c, 0) for c in cats_order]
    bars = ax.bar(range(4), counts, color=[cat_colors_d[c] for c in cats_order],
                  alpha=0.75, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(4))
    nice = ["Both\ncorrect", "Attn helps\n(Q side)", "Both\nwrong", "Attn\nhurts"]
    ax.set_xticklabels(nice, fontsize=7)
    ax.set_ylabel("Count", fontsize=9)
    ax.set_title("Category (MLP embed-only at Q)", fontsize=10, fontweight="bold")
    for bar, n in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                f"{n}\n({n/total:.1%})", ha="center", va="bottom", fontsize=7, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # (1,0): Failure rate by min_gap (query-side)
    ax = axes[1][0]
    gap_bins = [(1, 1), (2, 2), (3, 3), (4, 5), (6, 10), (11, 20), (21, 50)]
    helps_rates = []
    gap_labels_l = []
    for lo, hi in gap_bins:
        n_t = sum(1 for r in records if lo <= r["min_gap"] <= hi)
        n_h = sum(1 for r in records if lo <= r["min_gap"] <= hi and r["category"] == "attn_helps")
        if n_t > 0:
            helps_rates.append((n_h / n_t, n_t))
            gap_labels_l.append(f"{lo}" if lo == hi else f"{lo}-{hi}")
    if helps_rates:
        x = np.arange(len(helps_rates))
        ax.bar(x, [h[0] for h in helps_rates], color="#b2182b", alpha=0.75)
        ax.set_xticks(x)
        ax.set_xticklabels(gap_labels_l, fontsize=8)
        for i, (rate, n) in enumerate(helps_rates):
            ax.text(i, rate + 0.005, f"{rate:.1%}\n(n={n})", ha="center", fontsize=6.5)
    ax.set_xlabel("Min gap to neighboring sorted value", fontsize=9)
    ax.set_ylabel("Attn-helps rate (Q side)", fontsize=9)
    ax.set_title("Q-side failure rate by min gap", fontsize=10, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # (1,1): Failure rate by density
    ax = axes[1][1]
    density_bins = list(range(0, 7))
    fail_d = []
    for d in density_bins:
        n_t = sum(1 for r in records if r["n_within_5"] == d)
        n_h = sum(1 for r in records if r["n_within_5"] == d and r["category"] == "attn_helps")
        if n_t > 10:
            fail_d.append((d, n_h / n_t, n_t))
    if fail_d:
        x = np.arange(len(fail_d))
        ax.bar(x, [f[1] for f in fail_d], color="#b2182b", alpha=0.75)
        ax.set_xticks(x)
        ax.set_xticklabels([str(f[0]) for f in fail_d], fontsize=8)
        for i, (_, rate, n) in enumerate(fail_d):
            ax.text(i, rate + 0.005, f"{rate:.1%}\n(n={n})", ha="center", fontsize=6.5)
    ax.set_xlabel("# input numbers within ±5 of target", fontsize=9)
    ax.set_ylabel("Attn-helps rate (Q side)", fontsize=9)
    ax.set_title("Q-side failure rate by density", fontsize=10, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # (1,2): Failure rate by sorted position
    ax = axes[1][2]
    pos_bins = [(1, 4), (5, 8), (9, 12), (13, 16), (17, 20), (21, 24), (25, 28), (29, 31)]
    fail_p = []
    for lo, hi in pos_bins:
        n_t = sum(1 for r in records if lo <= r["sorted_pos"] <= hi)
        n_h = sum(1 for r in records if lo <= r["sorted_pos"] <= hi and r["category"] == "attn_helps")
        if n_t > 0:
            fail_p.append((f"{lo}-{hi}", n_h / n_t, n_t))
    if fail_p:
        x = np.arange(len(fail_p))
        ax.bar(x, [f[1] for f in fail_p], color="#b2182b", alpha=0.75)
        ax.set_xticks(x)
        ax.set_xticklabels([f[0] for f in fail_p], fontsize=7)
        for i, (_, rate, n) in enumerate(fail_p):
            ax.text(i, rate + 0.005, f"{rate:.1%}\n(n={n})", ha="center", fontsize=6.5)
    ax.set_xlabel("Sorted output position", fontsize=9)
    ax.set_ylabel("Attn-helps rate (Q side)", fontsize=9)
    ax.set_title("Q-side failure rate by position", fontsize=10, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # (1,3): Summary text
    ax = axes[1][3]
    ax.axis("off")
    lines = ["QUERY-SIDE ABLATION SUMMARY", "=" * 42, ""]
    lines.append(f"{'Condition':28s} {'Attn wt':>8s} {'Top-1':>8s}")
    lines.append("-" * 48)
    for cond in conditions:
        m_a = np.mean(attn_on_correct[cond])
        m_t = np.mean(top1_correct[cond])
        lines.append(f"{cond_labels[cond].replace(chr(10), ' '):28s} {m_a:8.4f} {m_t:7.1%}")
    lines.append("")
    lines.append(f"Categories (embed-only Q):")
    for c in cats_order:
        n = cat_counts.get(c, 0)
        lines.append(f"  {c:15s}: {n:6d} ({n/total:.1%})")

    ax.text(0.02, 0.98, "\n".join(lines), transform=ax.transAxes,
            fontsize=7.5, verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0", alpha=0.8))

    fig.suptitle(
        f"Query-side ablation: does L1 attn matter for queries?\n"
        f"k={block_size}, N={vocab_n}, {os.path.basename(ARGS.ckpt)}  |  {n_trials} trials",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("QUERY-SIDE ABLATION SUMMARY")
    print("=" * 60)
    print(f"\n{'Condition':28s} {'Attn wt':>10s} {'Top-1':>10s}")
    for cond in conditions:
        m_a = np.mean(attn_on_correct[cond])
        m_t = np.mean(top1_correct[cond])
        print(f"  {cond_labels[cond].replace(chr(10), ' '):26s}: {m_a:.4f}     {m_t:.1%}")

    print(f"\nCategories (embed-only Q, total={total}):")
    for c in cats_order:
        n = cat_counts.get(c, 0)
        print(f"  {c:15s}: {n:6d} ({n/total:.1%})")


if __name__ == "__main__":
    main()
