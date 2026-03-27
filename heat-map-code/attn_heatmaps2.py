#!/usr/bin/env python3
"""
Standalone script to generate attention probability heatmaps.

Supports both checkpoint formats:
  - grid-run: keys 'model', 'config' (vocab_size is base value)
  - train_single_80k: keys 'model_state_dict', 'model_config' (vocab_size = base + 1)

Usage:
    python attn_heatmaps2.py path/to/checkpoint.pt [--output-dir DIR]

If --output-dir is omitted, the plot is saved next to the checkpoint.
"""

import argparse
import sys
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _import_model():
    """Try grid-run model_analysis first, fall back to local."""
    grid_run = Path(__file__).resolve().parent.parent / "grid-run"
    if grid_run.exists():
        sys.path.insert(0, str(grid_run))
    from model_analysis import GPT, GPTConfig
    return GPT, GPTConfig


# ── Data helpers ──────────────────────────────────────────────────────────────

def make_generator(device: torch.device, seed: int) -> torch.Generator:
    try:
        g = torch.Generator(device=device.type)
    except Exception:
        g = torch.Generator()
    g.manual_seed(int(seed))
    return g


def get_batch(batch_size, length, device, *, vocab_n, generator=None):
    scores = torch.rand(batch_size, vocab_n, device=device, generator=generator)
    x = scores.topk(length, dim=1).indices.to(torch.long)
    vals = x.sort(dim=1).values
    sep = torch.full((batch_size, 1), vocab_n, device=device, dtype=torch.long)
    return torch.cat([x, sep, vals], dim=1)


# ── Axis labelling ────────────────────────────────────────────────────────────

def get_sorted_token_axis_info(idx_row, block_size):
    pre = idx_row[:block_size].to(torch.long)
    post = idx_row[block_size + 1:].to(torch.long)
    pre_sort_positions = torch.argsort(pre)
    pre_ranks = torch.empty(block_size, dtype=torch.long)
    pre_ranks[pre_sort_positions] = torch.arange(block_size, dtype=torch.long)
    row_order = list(pre_sort_positions.numpy()) + [block_size] + list(range(block_size + 1, 2 * block_size + 1))
    row_labels = [f'{pre[i].item()}(r{pre_ranks[i].item()})' for i in pre_sort_positions] \
                 + ['SEP'] \
                 + [f'{post[j].item()}' for j in range(block_size)]
    return {'row_order': row_order, 'row_labels': row_labels}


# ── Main plotting ─────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_attn_heatmap(model, block_size, vocab_n, output_dir, desc_str):
    gen = make_generator(DEVICE, 20260308)
    idx = get_batch(1, block_size, DEVICE, vocab_n=vocab_n, generator=gen)

    model(idx)

    n_layers = model.config.n_layers
    idx_row = idx[0].detach().cpu()
    sort_info = get_sorted_token_axis_info(idx_row, block_size)
    order = sort_info['row_order']
    labels = sort_info['row_labels']

    fig, axes = plt.subplots(1, n_layers, figsize=(8 * n_layers, 6), squeeze=False)
    for col_i in range(n_layers):
        ax = axes[0][col_i]
        probs_np = model.transformer.h[col_i].c_attn.attn.detach().cpu().numpy()
        probs_np = probs_np[np.ix_(order, order)]
        im = ax.imshow(probs_np, aspect='auto', interpolation='nearest', vmin=0)
        ax.set_title(f'Layer {col_i + 1} attention probs', fontsize=11)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel('Attended-to token (sorted by rank)', fontsize=9)
        ax.set_ylabel('Current token (sorted by rank)', fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f'Attention probability heatmaps\n{desc_str}', fontsize=13)
    plt.tight_layout()
    path = Path(output_dir) / 'attn_heatmaps.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")


def _remap_state_dict(sd):
    """Remap 1000k-checkpoint naming (attn/mlp) to model_analysis naming (c_attn/c_fc)."""
    new_sd = {}
    for key, val in sd.items():
        new_key = key
        for i in range(10):
            new_key = new_key.replace(f'transformer.h.{i}.attn.', f'transformer.h.{i}.c_attn.')
            new_key = new_key.replace(f'transformer.h.{i}.mlp.', f'transformer.h.{i}.c_fc.')
        new_sd[new_key] = val
    return new_sd


def main():
    GPT, GPTConfig = _import_model()

    parser = argparse.ArgumentParser(description="Generate attention heatmaps")
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint .pt file")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save the plot (default: same dir as checkpoint)")
    args = parser.parse_args()

    pt_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir) if args.output_dir else pt_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint: {pt_path}")
    artifact = torch.load(pt_path, map_location="cpu")

    needs_remap = False
    if "config" in artifact and "model" in artifact:
        cfg = artifact["config"]
        block_size = cfg["block_size"]
        vocab_n = cfg["vocab_size"]
        with_layer_norm = cfg.get("with_layer_norm", False)
        state_dict = artifact["model"]
    else:
        model_conf = artifact["model_config"]
        block_size = model_conf["block_size"]
        vocab_n = model_conf["vocab_size"] - 1
        with_layer_norm = model_conf.get("use_final_LN", False)
        state_dict = artifact["model_state_dict"]
        needs_remap = model_conf.get("max_seq_len", 0) > block_size * 4 + 1

    if needs_remap:
        state_dict = _remap_state_dict(state_dict)
        grid_wpe_size = block_size * 4 + 1
        if 'transformer.wpe.weight' in state_dict and state_dict['transformer.wpe.weight'].shape[0] > grid_wpe_size:
            state_dict['transformer.wpe.weight'] = state_dict['transformer.wpe.weight'][:grid_wpe_size]
        keys_to_drop = [k for k in state_dict if k.endswith('.c_attn.bias') and 'c_attn.c_attn' not in k]
        for k in keys_to_drop:
            del state_dict[k]
        if 'lm_head.weight' in state_dict:
            del state_dict['lm_head.weight']

    config = GPTConfig(block_size=block_size, vocab_size=vocab_n, with_layer_norm=with_layer_norm)
    model = GPT(config)
    model.load_state_dict(state_dict, strict=False)
    model.to(DEVICE).eval()

    desc_str = f"k={block_size} / N={vocab_n} / E={config.n_embd} / LN={int(with_layer_norm)}"

    print(f"Model: {desc_str} | device={DEVICE}")
    generate_attn_heatmap(model, block_size, vocab_n, output_dir, desc_str)


if __name__ == "__main__":
    main()
