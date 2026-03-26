#!/usr/bin/env python3
"""
Run hijack intervention experiment on layer 1 for a single checkpoint,
then produce breaking-rate, hijack-rate, and sample-count heatmaps.
"""
import os
import sys
import types
import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'grid-run'))
from model_analysis import GPT, GPTConfig

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BIN_SIZE = 8
N_BINS = 256 // BIN_SIZE
INTENSITY = 10.0
LAYER = 1


def remap_state_dict(sd):
    new_sd = {}
    for key, val in sd.items():
        new_key = key
        for i in range(10):
            new_key = new_key.replace(f'transformer.h.{i}.attn.', f'transformer.h.{i}.c_attn.')
            new_key = new_key.replace(f'transformer.h.{i}.mlp.', f'transformer.h.{i}.c_fc.')
        new_sd[new_key] = val
    return new_sd


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    mc = ckpt['model_config']
    vocab_size = mc['vocab_size'] - 1
    block_size = mc['block_size']
    with_layer_norm = mc.get('use_final_LN', True)

    config = GPTConfig(block_size=block_size, vocab_size=vocab_size,
                       with_layer_norm=with_layer_norm)
    model = GPT(config)

    sd = remap_state_dict(ckpt['model_state_dict'])
    grid_wpe_size = block_size * 4 + 1
    if 'transformer.wpe.weight' in sd and sd['transformer.wpe.weight'].shape[0] > grid_wpe_size:
        sd['transformer.wpe.weight'] = sd['transformer.wpe.weight'][:grid_wpe_size]
    keys_to_skip = [k for k in sd if k.endswith('.c_attn.bias') and 'c_attn.c_attn' not in k]
    for k in keys_to_skip:
        del sd[k]
    if 'lm_head.weight' in sd:
        del sd['lm_head.weight']

    model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    return model, config


def get_batch(vocab_size, block_size, device='cpu'):
    x = torch.randperm(vocab_size)[:block_size]
    vals, _ = torch.sort(x)
    return torch.cat((x, torch.tensor([vocab_size]), vals), dim=0).unsqueeze(0).to(device)


def compute_hijack_layer(model, config, device, layer, n_trials=2000):
    bs = config.block_size
    vs = config.vocab_size
    attn_module = model.transformer.h[layer].c_attn
    records = []

    for trial in range(n_trials):
        idx = get_batch(vs, bs, device)
        unsorted = idx[0, :bs]
        sorted_part = idx[0, bs + 1: 2 * bs + 1]

        with torch.no_grad():
            _, _ = model(idx)
        raw_attn = attn_module.raw_attn.clone()

        for p in range(bs - 1):
            location = bs + 1 + p
            current_num = sorted_part[p].item()
            correct_next = idx[0, location + 1].item()

            next_loc_in_unsorted = (unsorted == correct_next).nonzero(as_tuple=True)[0]
            if len(next_loc_in_unsorted) == 0:
                continue
            next_loc = next_loc_in_unsorted[0].item()
            main_attn_val = raw_attn[location, next_loc].item()

            candidates = [i for i in range(bs) if unsorted[i].item() != correct_next]
            if not candidates:
                continue

            boost_idx = candidates[torch.randint(len(candidates), (1,)).item()]
            boosted_number = unsorted[boost_idx].item()

            def make_new_forward(loc, bidx, mav):
                def new_forward(self_attn, x, layer_n=-1):
                    B, T, C = x.size()
                    qkv = self_attn.c_attn(x)
                    q, k, v = qkv.split(self_attn.n_embd, dim=2)
                    q = q.view(B, T, self_attn.n_heads, C // self_attn.n_heads).transpose(1, 2)
                    k = k.view(B, T, self_attn.n_heads, C // self_attn.n_heads).transpose(1, 2)
                    v = v.view(B, T, self_attn.n_heads, C // self_attn.n_heads).transpose(1, 2)
                    attn = q @ k.transpose(-1, -2) * 0.1 / (k.size(-1)) ** 0.5
                    attn[:, :, loc, bidx] = mav + INTENSITY
                    attn = attn.masked_fill(self_attn.bias[:, :, :T, :T] == 0, float('-inf'))
                    attn = F.softmax(attn, dim=-1)
                    y = attn @ v
                    y = y.transpose(1, 2).contiguous().view(B, T, C)
                    y = self_attn.c_proj(y)
                    return y
                return new_forward

            old_forward = attn_module.forward
            attn_module.forward = types.MethodType(
                make_new_forward(location, boost_idx, main_attn_val), attn_module)

            with torch.no_grad():
                logits, _ = model(idx)
            predicted = torch.argmax(logits, dim=-1)[0, location].item()

            attn_module.forward = old_forward
            records.append((current_num, boosted_number, predicted, correct_next))

        if (trial + 1) % 200 == 0:
            print(f"  Trial {trial+1}/{n_trials}, {len(records)} records so far", flush=True)

    return np.array(records, dtype=np.int32) if records else np.empty((0, 4), dtype=np.int32)


def plot_hijack_heatmaps(data, plot_dir, tag, layer):
    if len(data) == 0:
        print("No hijack data to plot!")
        return

    current = data[:, 0]; boosted = data[:, 1]
    predicted = data[:, 2]; correct = data[:, 3]
    broken = (predicted != correct).astype(np.float64)
    hijacked = (predicted == boosted).astype(np.float64)
    cur_bin = np.clip(current // BIN_SIZE, 0, N_BINS - 1)
    bst_bin = np.clip(boosted // BIN_SIZE, 0, N_BINS - 1)

    break_map = np.full((N_BINS, N_BINS), np.nan)
    hijack_map = np.full((N_BINS, N_BINS), np.nan)
    count_map = np.zeros((N_BINS, N_BINS), dtype=int)
    for cb in range(N_BINS):
        for bb in range(N_BINS):
            mask = (cur_bin == cb) & (bst_bin == bb)
            n = mask.sum()
            count_map[cb, bb] = n
            if n >= 5:
                break_map[cb, bb] = broken[mask].mean()
                hijack_map[cb, bb] = hijacked[mask].mean()

    tick_labels = [f'{i * BIN_SIZE}' for i in range(0, N_BINS, 4)]
    tick_positions = list(range(0, N_BINS, 4))

    for arr, cmap, label, fname in [
        (break_map, 'YlOrRd', 'Breaking Rate',
         f'hijack_breaking_rate_heatmap_layer{layer}.png'),
        (hijack_map, 'YlOrRd', 'Hijack Rate',
         f'hijack_hijack_rate_heatmap_layer{layer}.png'),
    ]:
        fig, ax = plt.subplots(figsize=(10, 8.5))
        im = ax.imshow(arr, aspect='auto', cmap=cmap, vmin=0, vmax=1,
                       interpolation='nearest', origin='lower')
        ax.set_xlabel('Intervened-toward Number (binned)', fontsize=12)
        ax.set_ylabel('Current Number (binned)', fontsize=12)
        title_map = {'Breaking Rate': f'Breaking Rate: P(pred \u2260 correct)',
                     'Hijack Rate': f'Hijack Rate: P(pred == intervened target)'}
        ax.set_title(f'{title_map[label]}\n{tag}  layer={layer}  intensity={INTENSITY}',
                     fontsize=12, fontweight='bold')
        ax.set_xticks(tick_positions); ax.set_xticklabels(tick_labels, fontsize=8)
        ax.set_yticks(tick_positions); ax.set_yticklabels(tick_labels, fontsize=8)
        plt.colorbar(im, ax=ax, label=label, shrink=0.85)
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, fname), dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {fname}")

    fig, ax = plt.subplots(figsize=(10, 8.5))
    im = ax.imshow(count_map, aspect='auto', cmap='viridis',
                   interpolation='nearest', origin='lower')
    ax.set_xlabel('Intervened-toward Number (binned)', fontsize=12)
    ax.set_ylabel('Current Number (binned)', fontsize=12)
    ax.set_title(f'Sample Count per (current, target) bin\n{tag}  layer={layer}  intensity={INTENSITY}',
                 fontsize=11, fontweight='bold')
    ax.set_xticks(tick_positions); ax.set_xticklabels(tick_labels, fontsize=8)
    ax.set_yticks(tick_positions); ax.set_yticklabels(tick_labels, fontsize=8)
    plt.colorbar(im, ax=ax, label='Count', shrink=0.85)
    fig.tight_layout()
    fname = f'hijack_sample_count_heatmap_layer{layer}.png'
    fig.savefig(os.path.join(plot_dir, fname), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}")


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ckpt_path = os.path.join(SCRIPT_DIR,
        'sortgpt_k16_methfixed_mlp1_L2_N256_E64_pos0_fln1_wd0p0_lr0p03_dseed1337_iseed1337__ckpt50000.pt')
    plot_dir = os.path.join(SCRIPT_DIR, 'outputs',
        'plots_V256_B16_LR3e-2_MI50000_E64_H1_L2_ds1337_is1337_ckpt50000')

    os.makedirs(plot_dir, exist_ok=True)

    print(f"Loading model from {os.path.basename(ckpt_path)} ...", flush=True)
    model, config = load_model(ckpt_path, device)
    print(f"Model loaded. block_size={config.block_size}, vocab_size={config.vocab_size}", flush=True)

    print(f"\nRunning hijack experiment on layer {LAYER} (2000 trials) ...", flush=True)
    data = compute_hijack_layer(model, config, device, layer=LAYER, n_trials=2000)
    print(f"Collected {len(data)} records", flush=True)

    tag = "V=256  B=16  lr=0.03  iters=50000  dseed=1337  iseed=1337"
    print(f"\nGenerating heatmap plots ...", flush=True)
    plot_hijack_heatmaps(data, plot_dir, tag, layer=LAYER)

    print("\nDone!")


if __name__ == '__main__':
    main()
