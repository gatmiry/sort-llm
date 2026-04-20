#!/usr/bin/env python3
"""
Generate hijack_comparison_3way plots for multiple gap sizes (3, 5, 10).

Three hijack types:
  MLP1 hijack  – force attn1→wrong key, recompute mlp1, feed fake mlp1 to MLP2 (attn2 normal)
  ATTN1 hijack – force attn1→wrong key, everything flows naturally
  ATTN2 hijack – force attn2→wrong key, mlp1 stays normal
"""
import os, sys, math, types, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'sortgpt_toolkit'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from model import DEVICE, load_model_from_checkpoint
from intervene import enable_attention_storage

CKPT = os.path.join(os.path.dirname(__file__), '..', 'new-grid', 'k32_N512',
                     'checkpoints', 'std0p01_iseed1__ckpt100000.pt')
OUTDIR = os.path.join(os.path.dirname(__file__), '..', 'new-grid', 'k32_N512', 'plots')
os.makedirs(OUTDIR, exist_ok=True)

plt.rcParams.update({
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'savefig.facecolor': 'white', 'axes.spines.top': False,
    'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': 0.2,
    'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 13,
})

print(f"Loading model from {CKPT} ...", flush=True)
model = load_model_from_checkpoint(CKPT)
enable_attention_storage(model)
block_size = model.config.block_size
vocab_n = model.config.vocab_size - 1
block0, block1 = model.transformer.h[0], model.transformer.h[1]
print(f"  block_size={block_size}, vocab_n={vocab_n}, device={DEVICE}", flush=True)

I_RANGES = [(0, 20), (20, 80), (80, 200), (200, 350), (350, vocab_n)]
COLORS = {
    (0, 20): '#e41a1c', (20, 80): '#ff7f00', (80, 200): '#4daf4a',
    (200, 350): '#377eb8', (350, vocab_n): '#984ea3',
}
N_TRIALS = 250


def generate_gap_batch(gap):
    """Sequence where consecutive sorted values differ by exactly `gap`."""
    max_start = vocab_n - gap * block_size
    if max_start <= 0:
        return None
    start = torch.randint(0, max_start, (1,)).item()
    vals = torch.arange(start, start + gap * block_size, gap)[:block_size]
    if len(vals) < block_size or vals[-1] >= vocab_n:
        return None
    perm = torch.randperm(block_size)
    unsorted = vals[perm]
    sorted_vals, _ = torch.sort(unsorted)
    sep = torch.tensor([vocab_n])
    return torch.cat((unsorted, sep, sorted_vals)).unsqueeze(0).to(DEVICE)


@torch.no_grad()
def precompute_block0(idx):
    """Shared block-0 quantities for a given batch."""
    B, T = idx.size()
    pos = model.transformer.wpe(model.pos_idx[:T])
    embed = model.transformer.wte(idx) + pos

    h0 = block0.ln_1(embed)
    qkv = block0.attn.c_attn(h0)
    ne = block0.attn.n_embd
    nh, hd = block0.attn.n_heads, block0.attn.head_dim
    q, k, v = qkv.split(ne, dim=2)
    q4 = q.view(B, T, nh, hd).transpose(1, 2)
    k4 = k.view(B, T, nh, hd).transpose(1, 2)
    v4 = v.view(B, T, nh, hd).transpose(1, 2)

    scale = 1.0 / math.sqrt(hd)
    raw_att = (q4 @ k4.transpose(-2, -1)) * scale
    causal = torch.triu(torch.ones(T, T, device=DEVICE, dtype=torch.bool), diagonal=1)
    att = raw_att.masked_fill(causal.unsqueeze(0).unsqueeze(0), float('-inf'))
    att = F.softmax(att, dim=-1)
    y = (att @ v4).transpose(1, 2).contiguous().view(B, T, ne)
    real_attn0 = block0.attn.c_proj(y)

    x_after_attn0 = embed + real_attn0
    real_mlp0 = block0.mlp(block0.ln_2(x_after_attn0))
    x_real = x_after_attn0 + real_mlp0

    h1_real = block1.ln_1(x_real)
    real_attn1 = block1.attn(h1_real)

    return dict(
        embed=embed, raw_att=raw_att, v4=v4, causal=causal,
        real_attn0=real_attn0, x_real=x_real,
        h1_real=h1_real, real_attn1=real_attn1, ne=ne,
    )


@torch.no_grad()
def mlp1_hijack(pre, idx, qpos, wpos):
    """Force attn1→wrong key, recompute mlp1, feed to MLP2. Attn2 uses real residual."""
    B, T = idx.size()
    raw = pre['raw_att'].clone()
    raw[:, :, qpos, :] = -1e9
    raw[:, :, qpos, wpos] = 20.0
    att = raw.masked_fill(pre['causal'].unsqueeze(0).unsqueeze(0), float('-inf'))
    att = F.softmax(att, dim=-1)
    y = (att @ pre['v4']).transpose(1, 2).contiguous().view(B, T, pre['ne'])
    forced_attn0 = block0.attn.c_proj(y)

    x_forced = pre['embed'] + forced_attn0
    forced_mlp0 = block0.mlp(block0.ln_2(x_forced))

    x_mod = pre['x_real'].clone()
    x_mod[0, qpos] = (pre['embed'][0, qpos]
                       + pre['real_attn0'][0, qpos]
                       + forced_mlp0[0, qpos])

    x_after = x_mod + pre['real_attn1']
    mlp_out = block1.mlp(block1.ln_2(x_after))
    final = model.transformer.ln_f(x_after + mlp_out)
    logits = model.lm_head(final)
    return logits[0, qpos].argmax().item()


@torch.no_grad()
def attn_hijack(idx, layer, qpos, wpos):
    """Full-model forward with forced attention at `layer`."""
    attn_mod = model.transformer.h[layer].attn
    old_fwd = attn_mod.forward

    def new_fwd(self_attn, x):
        B, T, C = x.size()
        qkv = self_attn.c_attn(x)
        q, k, v = qkv.split(self_attn.n_embd, dim=2)
        nh, hd = self_attn.n_heads, self_attn.head_dim
        q = q.view(B, T, nh, hd).transpose(1, 2)
        k = k.view(B, T, nh, hd).transpose(1, 2)
        v = v.view(B, T, nh, hd).transpose(1, 2)
        sc = 1.0 / math.sqrt(hd)
        a = (q @ k.transpose(-2, -1)) * sc
        a[:, :, qpos, :] = -1e9
        a[:, :, qpos, wpos] = 20.0
        cm = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        a = a.masked_fill(cm.unsqueeze(0).unsqueeze(0), float('-inf'))
        a = F.softmax(a, dim=-1)
        y = (a @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self_attn.c_proj(y)

    attn_mod.forward = types.MethodType(new_fwd, attn_mod)
    logits, _ = model(idx, block_size=block_size, return_full_logits=True)
    attn_mod.forward = old_fwd
    return logits[0, qpos].argmax().item()


def run_experiment(gap, offsets, n_trials):
    results = {ht: {off: {ir: [] for ir in I_RANGES} for off in offsets}
               for ht in ['mlp1', 'attn1', 'attn2']}
    t0 = time.time()

    for trial in range(n_trials):
        idx = generate_gap_batch(gap)
        if idx is None:
            continue

        sorted_vals = idx[0, block_size + 1:].cpu().numpy()
        unsorted = idx[0, :block_size].cpu().numpy()
        val_to_pos = {int(unsorted[p]): p for p in range(block_size)}

        pre = precompute_block0(idx)

        for p in range(block_size - 1):
            qpos = block_size + 1 + p
            cval = int(sorted_vals[p])
            tval = int(sorted_vals[p + 1])

            ir = None
            for lo, hi in I_RANGES:
                if lo <= cval < hi:
                    ir = (lo, hi)
                    break
            if ir is None:
                continue

            for off in offsets:
                wval = cval + off
                if wval == tval or wval >= vocab_n or wval not in val_to_pos:
                    continue
                wpos = val_to_pos[wval]

                pred = mlp1_hijack(pre, idx, qpos, wpos)
                results['mlp1'][off][ir].append(pred == wval)

                pred = attn_hijack(idx, 1, qpos, wpos)
                results['attn2'][off][ir].append(pred == wval)

                pred = attn_hijack(idx, 0, qpos, wpos)
                results['attn1'][off][ir].append(pred == wval)

        if (trial + 1) % 25 == 0:
            el = time.time() - t0
            eta = el / (trial + 1) * (n_trials - trial - 1)
            print(f"    gap={gap}: {trial+1}/{n_trials}  "
                  f"({el:.0f}s elapsed, ~{eta:.0f}s remaining)", flush=True)

    return results


def plot_hijack(gap, results, offsets):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    titles = {
        'mlp1': 'MLP1 hijack\n(fake mlp1 \u2192 MLP2, attn2 normal)',
        'attn1': 'ATTN1 hijack\n(force attn1, everything flows)',
        'attn2': 'ATTN2 hijack\n(force attn2, mlp1 normal)',
    }

    for ax, ht in zip(axes, ['mlp1', 'attn1', 'attn2']):
        for ir in I_RANGES:
            rates, valid = [], []
            for off in offsets:
                vals = results[ht][off][ir]
                if len(vals) >= 10:
                    rates.append(100 * np.mean(vals))
                    valid.append(off)
            if valid:
                ax.plot(valid, rates, 'o-', color=COLORS[ir],
                        linewidth=1.8, markersize=5, alpha=0.85,
                        label=f'{ir[0]}-{ir[1]}')
        ax.set_xlabel('Offset (hijack to i+offset)')
        ax.set_ylabel('Hijack success rate (%)')
        ax.set_title(titles[ht])
        ax.legend(fontsize=9)
        ax.set_ylim(-5, 105)

    fig.suptitle(f'Comparison of three hijack interventions (gap={gap}, N512_s1)',
                 fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    out = os.path.join(OUTDIR, f'hijack_comparison_3way_gap{gap}.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved {out}", flush=True)


if __name__ == '__main__':
    gaps_offsets = {
        3:  [6, 9, 12, 15, 18, 21],
        5:  [10, 15, 20, 25, 30, 35],
        10: [20, 30, 40, 50, 60],
    }
    for gap, offsets in gaps_offsets.items():
        print(f"\n{'='*60}\nGap {gap}  |  offsets = {offsets}\n{'='*60}", flush=True)
        results = run_experiment(gap, offsets, N_TRIALS)
        plot_hijack(gap, results, offsets)

        for ht in ['mlp1', 'attn1', 'attn2']:
            for ir in I_RANGES:
                counts = [len(results[ht][o][ir]) for o in offsets]
                print(f"  {ht} {ir}: counts = {counts}")

    print("\nAll done.", flush=True)
