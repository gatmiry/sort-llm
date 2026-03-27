#!/usr/bin/env python3
"""
Generate attention heatmaps for trained SortGPT models.

Two types:
  1. Positional heatmap (33×33): full attention pattern for a single sample,
     with unsorted tokens reordered by value.
  2. Averaged heatmap (256×256): sorted→unsorted attention averaged over
     1000 samples, indexed by token value.

Usage:
    CUDA_VISIBLE_DEVICES=X python plot_heatmaps.py \\
        --checkpoints ckpt1.pt ckpt2.pt ... \\
        --labels "seed=1501" "seed=1502" ... \\
        --output-dir ./plots

    # Or point at a run directory to auto-discover final checkpoints:
    CUDA_VISIBLE_DEVICES=X python plot_heatmaps.py \\
        --run-dir ./my_run --output-dir ./plots
"""

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from model import DEVICE, GPT, GPTConfig, make_generator, float_token, get_batch as _model_get_batch


# ── Batch generation (simplified for heatmaps) ──────────────────────────────

def get_batch_simple(batch_size, block_size, vocab_n, device, generator):
    scores = torch.rand(batch_size, vocab_n, device=device, generator=generator)
    x = scores.topk(block_size, dim=1).indices.to(torch.long)
    vals = x.sort(dim=1).values
    sep = torch.full((batch_size, 1), vocab_n, device=device, dtype=torch.long)
    return torch.cat([x, sep, vals], dim=1)


# ── Attention extraction ─────────────────────────────────────────────────────

def compute_attn_probs(attn_module, x):
    B, T, C = x.size()
    qkv = attn_module.c_attn(x)
    q, k, v = qkv.split(attn_module.n_embd, dim=2)
    q = q.view(B, T, attn_module.n_heads, attn_module.head_dim).transpose(1, 2)
    k = k.view(B, T, attn_module.n_heads, attn_module.head_dim).transpose(1, 2)
    scale = 1.0 / math.sqrt(attn_module.head_dim)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    causal_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(causal_mask, torch.finfo(scores.dtype).min)
    return torch.softmax(scores, dim=-1)


@torch.no_grad()
def get_all_layer_probs(model, idx):
    B, T = idx.size()
    x = model.transformer.wte(idx) if model.config.without_pos else (
        model.transformer.wte(idx) + model.transformer.wpe(model.pos_idx[:T]))
    result = {}
    for li, block in enumerate(model.transformer.h):
        ln1_x = block.ln_1(x)
        probs = compute_attn_probs(block.attn, ln1_x)
        result[li] = probs
        x = x + block.attn(ln1_x)
        if block.mlp:
            x = x + block.mlp(block.ln_2(x))
    return result


def get_sorted_token_axis_info(idx_row, block_size):
    """Reorder indices and labels so unsorted tokens appear in value-sorted order."""
    pre = idx_row[:block_size].to(torch.long)
    post = idx_row[block_size + 1:].to(torch.long)
    pre_sort_positions = torch.argsort(pre)
    pre_ranks = torch.empty(block_size, dtype=torch.long)
    pre_ranks[pre_sort_positions] = torch.arange(block_size, dtype=torch.long)
    row_order = (list(pre_sort_positions.numpy()) + [block_size]
                 + list(range(block_size + 1, 2 * block_size + 1)))
    row_labels = ([f'{pre[i].item()}(r{pre_ranks[i].item()})' for i in pre_sort_positions]
                  + ['SEP']
                  + [f'{post[j].item()}' for j in range(block_size)])
    return row_order, row_labels


# ── Averaged heatmap accumulation ────────────────────────────────────────────

def accumulate_avg_heatmap(model, block_size, vocab_n, n_samples=1000, batch_size=64, seed=20260308):
    n_layers = model.n_layers
    attn_sum = [torch.zeros(vocab_n, vocab_n, device='cpu') for _ in range(n_layers)]
    count = [torch.zeros(vocab_n, vocab_n, device='cpu') for _ in range(n_layers)]
    gen = torch.Generator(device=DEVICE.type)
    gen.manual_seed(seed)
    total_batches = (n_samples + batch_size - 1) // batch_size
    for b_idx in tqdm(range(total_batches), desc='accumulating', leave=False):
        bs = min(batch_size, n_samples - b_idx * batch_size)
        idx = get_batch_simple(bs, block_size, vocab_n, DEVICE, gen)
        unsorted_tokens = idx[:, :block_size]
        sorted_tokens = idx[:, block_size + 1:]
        all_probs = get_all_layer_probs(model, idx)
        for li in range(n_layers):
            probs = all_probs[li].mean(dim=1)
            sub_probs = probs[:, block_size + 1:2 * block_size + 1, :block_size]
            sub_probs_cpu = sub_probs.detach().cpu()
            sorted_tok_cpu = sorted_tokens.detach().cpu().long()
            unsorted_tok_cpu = unsorted_tokens.detach().cpu().long()
            for b in range(bs):
                s_vals = sorted_tok_cpu[b]
                u_vals = unsorted_tok_cpu[b]
                p = sub_probs_cpu[b]
                for qi in range(block_size):
                    sv = s_vals[qi].item()
                    for ki in range(block_size):
                        uv = u_vals[ki].item()
                        attn_sum[li][sv, uv] += p[qi, ki].item()
                        count[li][sv, uv] += 1.0
    avg = []
    for li in range(n_layers):
        mask = count[li] > 0
        h = torch.zeros_like(attn_sum[li])
        h[mask] = attn_sum[li][mask] / count[li][mask]
        avg.append(h.numpy())
    return avg


# ── Load model from checkpoint ───────────────────────────────────────────────

def load_model(ckpt_path):
    artifact = torch.load(ckpt_path, map_location='cpu')
    cfg = GPTConfig(**artifact['model_config'])
    model = GPT(cfg)
    model.load_state_dict(artifact['model_state_dict'])
    return model.to(DEVICE).eval()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate attention heatmaps")
    parser.add_argument("--checkpoints", type=str, nargs="+",
                        help="Checkpoint .pt files to plot")
    parser.add_argument("--labels", type=str, nargs="+",
                        help="Labels for each checkpoint (must match --checkpoints length)")
    parser.add_argument("--run-dir", type=str,
                        help="Auto-discover checkpoints from a run directory")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save plots")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--vocab-n", type=int, default=256)
    parser.add_argument("--n-samples", type=int, default=1000,
                        help="Samples for averaged heatmap")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    BLOCK_SIZE = args.block_size
    VOCAB_N = args.vocab_n
    T = 2 * BLOCK_SIZE + 1

    # Discover checkpoints
    if args.run_dir:
        run_dir = Path(args.run_dir)
        ckpt_dir = run_dir / "checkpoints"
        if not ckpt_dir.exists():
            ckpt_dir = run_dir / "final_models"
        ckpt_files = sorted(ckpt_dir.glob("*.pt"))
        # Only take final or latest checkpoint per seed
        seed_to_ckpt = {}
        for f in ckpt_files:
            m = re.search(r'iseed(\d+)', f.stem)
            if m:
                seed = int(m.group(1))
                seed_to_ckpt[seed] = f  # last one wins (sorted order)
        ckpt_paths = list(seed_to_ckpt.values())
        labels = [f.stem.split("__")[0] for f in ckpt_paths]
    else:
        ckpt_paths = [Path(p) for p in args.checkpoints]
        labels = args.labels or [p.stem for p in ckpt_paths]

    print(f"Found {len(ckpt_paths)} checkpoints")

    # Fixed sample for positional heatmaps
    pos_gen = torch.Generator(device=DEVICE.type)
    pos_gen.manual_seed(20260308)
    pos_idx = get_batch_simple(1, BLOCK_SIZE, VOCAB_N, DEVICE, pos_gen)
    pos_idx_row = pos_idx[0].detach().cpu()
    order, axis_labels = get_sorted_token_axis_info(pos_idx_row, BLOCK_SIZE)

    all_pos = {}  # label -> {layer: np}
    all_avg = {}  # label -> [layer0, layer1]

    for ckpt_path, label in zip(ckpt_paths, labels):
        print(f"\n[{label}] {ckpt_path.name}")
        model = load_model(ckpt_path)
        n_layers = model.n_layers

        # Positional heatmap
        attn_probs = get_all_layer_probs(model, pos_idx)
        pos_maps = {}
        for li in range(n_layers):
            probs_np = attn_probs[li][0].mean(dim=0).cpu().numpy()
            pos_maps[li] = probs_np[np.ix_(order, order)]
        all_pos[label] = pos_maps

        # Averaged heatmap
        print(f"  Computing averaged heatmap ({args.n_samples} samples)...")
        avg_maps = accumulate_avg_heatmap(model, BLOCK_SIZE, VOCAB_N, n_samples=args.n_samples)
        all_avg[label] = avg_maps

        del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    n_models = len(labels)
    n_layers = len(list(all_pos.values())[0])

    # ── Combined positional grid ──────────────────────────────────────────
    for li in range(n_layers):
        fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 6), squeeze=False)
        for col, label in enumerate(labels):
            ax = axes[0][col]
            im = ax.imshow(all_pos[label][li], aspect='auto', interpolation='nearest', vmin=0)
            ax.set_title(label, fontsize=10)
            ax.set_xticks(np.arange(len(axis_labels)))
            ax.set_xticklabels(axis_labels, rotation=90, fontsize=4)
            ax.set_yticks(np.arange(len(axis_labels)))
            ax.set_yticklabels(axis_labels, fontsize=4)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle(f'Layer {li+1} — Positional attention ({T}x{T})', fontsize=13, fontweight='bold')
        plt.tight_layout()
        path = output_dir / f"positional_heatmap_layer{li+1}.png"
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {path}")

    # ── Combined averaged grid ────────────────────────────────────────────
    for li in range(n_layers):
        fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 6), squeeze=False)
        for col, label in enumerate(labels):
            ax = axes[0][col]
            im = ax.imshow(all_avg[label][li], aspect='auto', interpolation='nearest', vmin=0)
            ax.set_title(label, fontsize=10)
            ax.set_xlabel('Unsorted token value', fontsize=9)
            ax.set_ylabel('Sorted token value', fontsize=9)
            ticks = list(range(0, VOCAB_N, 64))
            ax.set_xticks(ticks); ax.set_xticklabels(ticks, fontsize=7)
            ax.set_yticks(ticks); ax.set_yticklabels(ticks, fontsize=7)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle(f'Layer {li+1} — Averaged attention: sorted→unsorted ({args.n_samples} samples)',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        path = output_dir / f"averaged_heatmap_layer{li+1}.png"
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
