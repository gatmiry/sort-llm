#!/usr/bin/env python3
"""
For inputs with 6 consecutive numbers i..i+5 in sorted output:
- Query position = sorted position of i+1 (predicting i+2)
- Intervene Layer 1 pre-softmax logits: set the logit for each target
  t ∈ {i,..,i+5} to (correct_key_logit + intensity), where intensity=0
  means equal to the correct key.
- Plot resulting Layer 1 and Layer 2 attention bars at query_pos.

Grid: rows = samples, columns = intervention targets.
"""

import os, sys, random, math, argparse, types
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "sort-llm", "sortgpt_toolkit"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from model import DEVICE, load_model_from_checkpoint
from intervene import enable_attention_storage, disable_attention_storage

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", required=True)
parser.add_argument("--output", default=None)
parser.add_argument("--n-samples", type=int, default=8)
parser.add_argument("--consec-len", type=int, default=6)
parser.add_argument("--threshold", type=float, default=0.05)
parser.add_argument("--intensity", type=float, default=0.0)
parser.add_argument("--layer", type=int, default=0, help="Layer to intervene on")
ARGS = parser.parse_args()


def generate_consecutive_batch(block_size, vocab_n, consec_len, device):
    while True:
        i_start = random.randint(0, vocab_n - consec_len)
        consec = list(range(i_start, i_start + consec_len))
        remaining_pool = [v for v in range(vocab_n) if v not in consec]
        if len(remaining_pool) < block_size - consec_len:
            continue
        others = random.sample(remaining_pool, block_size - consec_len)
        tokens = consec + others
        random.shuffle(tokens)
        x = torch.tensor(tokens, dtype=torch.long, device=device)
        vals = x.sort().values
        sep = torch.tensor([vocab_n], dtype=torch.long, device=device)
        idx = torch.cat([x, sep, vals]).unsqueeze(0)
        sorted_vals = vals.tolist()
        for j in range(len(sorted_vals) - consec_len + 1):
            if sorted_vals[j:j + consec_len] == consec:
                return idx, i_start


@torch.no_grad()
def run_with_logit_intervention(model, idx, block_size, query_pos, target_input_pos,
                                correct_input_pos, intensity, interv_layer):
    """
    Run model with pre-softmax logit intervention on `interv_layer`:
    At query_pos, set logit for target_input_pos = logit(correct_input_pos) + intensity.
    Returns post-softmax attention probs for all layers.
    """
    attn_mod = model.transformer.h[interv_layer].attn

    # First, run clean to get the baseline logit for the correct key
    model(idx, block_size=block_size)
    raw_attn_baseline = attn_mod.raw_attn.clone()
    correct_logit = raw_attn_baseline[query_pos, correct_input_pos].item()

    if target_input_pos is None:
        # No intervention — return the clean attention
        layer_probs = []
        for block in model.transformer.h:
            layer_probs.append(block.attn.attn.clone())
        return layer_probs

    # Intervened forward: patch the interv_layer attention
    old_forward = attn_mod.forward

    def intervened_forward(self_attn, x):
        B, T, C = x.size()
        qkv = self_attn.c_attn(x)
        q, k, v = qkv.split(self_attn.n_embd, dim=2)
        q = q.view(B, T, self_attn.n_heads, self_attn.head_dim).transpose(1, 2)
        k = k.view(B, T, self_attn.n_heads, self_attn.head_dim).transpose(1, 2)
        v = v.view(B, T, self_attn.n_heads, self_attn.head_dim).transpose(1, 2)

        scale = 1.0 / math.sqrt(self_attn.head_dim)
        att = (q @ k.transpose(-2, -1)) * scale

        # Set target's logit = correct_logit + intensity
        att[:, :, query_pos, target_input_pos] = correct_logit + intensity

        causal = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        att = att.masked_fill(causal.unsqueeze(0).unsqueeze(0), float("-inf"))

        self_attn.raw_attn = att.clone().detach().squeeze(0).squeeze(0)
        att = F.softmax(att, dim=-1)
        self_attn.attn = att.clone().detach().squeeze(0).squeeze(0)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self_attn.c_proj(y)

    attn_mod.forward = types.MethodType(intervened_forward, attn_mod)
    model(idx, block_size=block_size)
    attn_mod.forward = old_forward

    layer_probs = []
    for block in model.transformer.h:
        layer_probs.append(block.attn.attn.clone())
    return layer_probs


@torch.no_grad()
def main():
    model = load_model_from_checkpoint(ARGS.ckpt)
    block_size = model.config.block_size
    vocab_n = model.config.vocab_size - 1
    consec_len = ARGS.consec_len
    intensity = ARGS.intensity

    output_path = ARGS.output or os.path.join(
        os.path.dirname(ARGS.ckpt), "..", "plots", "intervened_consecutive.png"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    enable_attention_storage(model)

    n_samples = ARGS.n_samples
    intervention_labels = ["No interv."] + [f"→ i+{d}" for d in range(consec_len)]
    n_interventions = len(intervention_labels)

    fig, axes = plt.subplots(n_samples, n_interventions,
                             figsize=(2.8 * n_interventions, 2.8 * n_samples),
                             squeeze=False)

    sample_i = 0
    attempts = 0
    while sample_i < n_samples and attempts < n_samples * 50:
        attempts += 1
        idx, i_start = generate_consecutive_batch(block_size, vocab_n, consec_len, DEVICE)
        target_val = i_start + 1

        sorted_part = idx[0, block_size + 1: 2 * block_size + 1]
        positions = (sorted_part == target_val).nonzero(as_tuple=True)[0]
        if len(positions) == 0:
            continue
        sorted_pos = positions[0].item()
        query_pos = block_size + 1 + sorted_pos

        unsorted_tokens = idx[0, :block_size].cpu().numpy()

        # The correct next number to predict at query_pos is i+2
        correct_next = i_start + 2
        correct_hits = (idx[0, :block_size] == correct_next).nonzero(as_tuple=True)[0]
        if len(correct_hits) == 0:
            continue
        correct_input_pos = correct_hits[0].item()

        # Find input positions for all consecutive values
        consec_input_positions = {}
        for d in range(consec_len):
            val = i_start + d
            hits = (idx[0, :block_size] == val).nonzero(as_tuple=True)[0]
            if len(hits) > 0:
                consec_input_positions[d] = hits[0].item()
        if len(consec_input_positions) < consec_len:
            continue

        consec_values = list(range(i_start, i_start + consec_len))
        bar_offsets = np.arange(consec_len)
        bar_width = 0.35

        for col, interv_label in enumerate(intervention_labels):
            ax = axes[sample_i][col]

            if col == 0:
                target_pos = None
            else:
                d = col - 1
                target_pos = consec_input_positions[d]

            layer_probs = run_with_logit_intervention(
                model, idx, block_size, query_pos, target_pos,
                correct_input_pos, intensity, ARGS.layer,
            )

            consec_pos_set = set(consec_input_positions[d] for d in range(consec_len))

            for layer_idx, color, shift in [
                (0, "#2166ac", -bar_width / 2),
                (1, "#b2182b", bar_width / 2),
            ]:
                attn = layer_probs[layer_idx]  # (T, T) post-softmax
                attn_full = attn[query_pos].cpu().numpy()
                attn_row = attn_full[:block_size]

                scores_consec = []
                for val in consec_values:
                    inp_pos = consec_input_positions[val - i_start]
                    scores_consec.append(attn_row[inp_pos])

                ax.bar(bar_offsets + shift, scores_consec,
                       width=bar_width, color=color, alpha=0.7)

            # Compute L2 attention breakdown outside the 6 consecutive numbers
            l2_attn = layer_probs[1][query_pos].cpu().numpy()
            consec_sum = sum(l2_attn[consec_input_positions[d]] for d in range(consec_len))
            # Other unsorted tokens (non-consecutive)
            other_unsorted_sum = 0.0
            other_unsorted_tokens = []
            for p in range(block_size):
                if p not in consec_pos_set:
                    score = l2_attn[p]
                    if score > 0.01:
                        other_unsorted_tokens.append((unsorted_tokens[p], score))
                    other_unsorted_sum += score
            # Sorted region + separator
            sorted_sum = l2_attn[block_size:].sum()

            # Build annotation text
            ann_lines = [f"L2 on i..i+{consec_len-1}: {consec_sum:.0%}"]
            ann_lines.append(f"other unsort: {other_unsorted_sum:.0%}")
            if other_unsorted_tokens:
                other_unsorted_tokens.sort(key=lambda x: -x[1])
                tok_strs = [f"{int(v)}:{s:.2f}" for v, s in other_unsorted_tokens[:4]]
                ann_lines.append("  " + ", ".join(tok_strs))
            ann_lines.append(f"sorted+sep: {sorted_sum:.0%}")
            ann_text = "\n".join(ann_lines)

            ax.text(0.98, 0.98, ann_text, transform=ax.transAxes,
                    fontsize=4.5, verticalalignment="top", horizontalalignment="right",
                    fontfamily="monospace",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="wheat", alpha=0.5))

            ax.set_xticks(bar_offsets)
            ax.set_xticklabels([f"i+{d}" for d in range(consec_len)], fontsize=6)
            ax.tick_params(labelsize=6)
            ax.set_ylim(0, None)
            ax.grid(True, axis="y", alpha=0.1, linestyle=":")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            if sample_i == 0:
                ax.set_title(interv_label, fontsize=9, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"i={i_start}", fontsize=8, fontweight="bold")
            if sample_i == n_samples - 1:
                ax.set_xlabel("token value", fontsize=7)

        print(f"  Sample {sample_i + 1}/{n_samples} (i={i_start})")
        sample_i += 1

    disable_attention_storage(model)

    fig.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color="#2166ac", alpha=0.7),
            plt.Rectangle((0, 0), 1, 1, color="#b2182b", alpha=0.7),
        ],
        labels=[f"Layer {ARGS.layer + 1}", f"Layer 2"],
        loc="upper right", fontsize=9, bbox_to_anchor=(0.99, 0.99),
    )
    fig.suptitle(
        f"Pre-softmax logit intervention (intensity={intensity}) on Layer {ARGS.layer + 1}\n"
        f"Attn at sorted pos of i+1  |  k={block_size}, N={vocab_n}, "
        f"{os.path.basename(ARGS.ckpt)}",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
