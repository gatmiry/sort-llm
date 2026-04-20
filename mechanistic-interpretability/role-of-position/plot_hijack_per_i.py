#!/usr/bin/env python3
"""
Per-value-of-i hijack comparison (3-way) for i = 0..10, configurable gap.

For each value of i, generates many controlled-gap batches, finds positions
where current sorted value == i, and runs all three hijack types on the SAME
instance/position:
  MLP1 hijack  – force attn1→wrong key, recompute mlp1, attn2 uses real residual
  ATTN1 hijack – force attn1→wrong key, everything flows naturally
  ATTN2 hijack – force attn2→wrong key, mlp1 stays normal

Output: 11-row × 3-column grid (one row per i, columns = MLP1 / ATTN1 / ATTN2).

Usage:
  python plot_hijack_per_i.py           # gap=1 (default)
  python plot_hijack_per_i.py --gap 3   # gap=3
"""
import os, sys, math, types, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'sortgpt_toolkit'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from model import DEVICE, load_model_from_checkpoint
from intervene import enable_attention_storage

parser = argparse.ArgumentParser()
parser.add_argument('--gap', type=int, default=1)
parser.add_argument('--i-range', type=str, default=None,
                    help='e.g. "10,30" for i=10..30')
parser.add_argument('--fine-offsets', action='store_true',
                    help='Use offsets 1..15 with random batches (not controlled-gap)')
parser.add_argument('--offsets', type=str, default=None,
                    help='Comma-separated custom offsets, e.g. "5,7,9,11,13"')
parser.add_argument('--group-avg', type=str, default=None,
                    help='Group i values and average, e.g. "1-3,4-6,7-9,10-12"')
parser.add_argument('--ckpt', type=str, default=None,
                    help='Path to checkpoint file (overrides default)')
parser.add_argument('--out-tag', type=str, default=None,
                    help='Extra tag appended to output filename')
parser.add_argument('--max-batches', type=int, default=None,
                    help='Override max batches (useful for large i ranges)')
parser.add_argument('--save-data', type=str, default=None,
                    help='Save per-offset rates to JSON file at this path')
ARGS = parser.parse_args()
GAP = ARGS.gap

if ARGS.ckpt:
    CKPT = ARGS.ckpt
else:
    CKPT = os.path.join(os.path.dirname(__file__), '..', '..', 'new-grid', 'k32_N512',
                         'checkpoints', 'std0p01_iseed1__ckpt100000.pt')
OUTDIR = os.path.join(os.path.dirname(__file__), 'plots')
os.makedirs(OUTDIR, exist_ok=True)

USE_RANDOM = ARGS.fine_offsets or ARGS.offsets is not None

GROUPS = None
if ARGS.group_avg:
    GROUPS = []
    for part in ARGS.group_avg.split(','):
        lo_g, hi_g = map(int, part.split('-'))
        GROUPS.append(list(range(lo_g, hi_g + 1)))
    I_VALUES = sorted({i for g in GROUPS for i in g})
elif ARGS.i_range:
    lo, hi = map(int, ARGS.i_range.split(','))
    I_VALUES = list(range(lo, hi + 1))
else:
    I_VALUES = list(range(11))
MIN_SAMPLES = 10 if len(I_VALUES) > 100 else 200
if ARGS.max_batches is not None:
    MAX_BATCHES = ARGS.max_batches
else:
    MAX_BATCHES = 300000 if USE_RANDOM else 30000

if ARGS.offsets is not None:
    OFFSETS = [int(x) for x in ARGS.offsets.split(',')]
elif ARGS.fine_offsets:
    OFFSETS = list(range(1, 16))
else:
    OFFSETS = sorted([m * GAP for m in range(1, 8)])

plt.rcParams.update({
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'savefig.facecolor': 'white', 'axes.spines.top': False,
    'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': 0.2,
    'font.size': 10,
})

print(f"Loading model ...", flush=True)
model = load_model_from_checkpoint(CKPT)
enable_attention_storage(model)
block_size = model.config.block_size
vocab_n = model.config.vocab_size - 1
block0, block1 = model.transformer.h[0], model.transformer.h[1]
print(f"  block_size={block_size}, vocab_n={vocab_n}", flush=True)


def generate_gap_batch():
    """Random batch where consecutive sorted values differ by exactly GAP."""
    max_start = vocab_n - GAP * block_size
    if max_start <= 0:
        return None
    start = torch.randint(0, max_start + 1, (1,)).item()
    vals = torch.arange(start, start + GAP * block_size, GAP)[:block_size]
    if len(vals) < block_size or vals[-1] >= vocab_n:
        return None
    perm = torch.randperm(block_size)
    unsorted = vals[perm]
    sorted_vals, _ = torch.sort(unsorted)
    sep = torch.tensor([vocab_n])
    return torch.cat((unsorted, sep, sorted_vals)).unsqueeze(0).to(DEVICE)


def generate_random_batch():
    """Standard random batch (no gap control)."""
    x = torch.randperm(vocab_n)[:block_size]
    vals, _ = torch.sort(x)
    sep = torch.tensor([vocab_n])
    return torch.cat((x, sep, vals)).unsqueeze(0).to(DEVICE)


def generate_targeted_batch(cval):
    """Random batch guaranteeing gap=GAP at cval, with offset targets present."""
    if cval + GAP >= vocab_n or cval < 0:
        return None
    forbidden = set(range(cval + 1, cval + GAP))
    required = {cval, cval + GAP}
    for off in OFFSETS:
        wval = cval + off
        if 0 <= wval < vocab_n and wval not in forbidden and wval != cval + GAP:
            required.add(wval)
    if len(required) > block_size:
        required = {cval, cval + GAP}
        for off in OFFSETS:
            wval = cval + off
            if 0 <= wval < vocab_n and wval not in forbidden and wval != cval + GAP:
                required.add(wval)
                if len(required) >= block_size:
                    break
    available = [v for v in range(vocab_n) if v not in required and v not in forbidden]
    remaining = block_size - len(required)
    if remaining < 0 or remaining > len(available):
        return None
    fill = np.random.choice(available, size=remaining, replace=False).tolist()
    all_vals = sorted(list(required) + fill)
    vals = torch.tensor(all_vals)
    perm = torch.randperm(block_size)
    unsorted = vals[perm]
    sep = torch.tensor([vocab_n])
    return torch.cat((unsorted, sep, vals)).unsqueeze(0).to(DEVICE)


@torch.no_grad()
def precompute_block0(idx):
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
def mlp1_and_attn2_hijack(pre, idx, qpos, wpos):
    """MLP1 hijack + force attn2 to same wrong key simultaneously."""
    B, T = idx.size()
    # Force attn1 → wrong key, recompute mlp1
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

    # Now also force attn2 → same wrong key
    h1 = block1.ln_1(x_mod)
    ne = block1.attn.n_embd
    nh, hd = block1.attn.n_heads, block1.attn.head_dim
    qkv = block1.attn.c_attn(h1)
    q, k, v = qkv.split(ne, dim=2)
    q4 = q.view(B, T, nh, hd).transpose(1, 2)
    k4 = k.view(B, T, nh, hd).transpose(1, 2)
    v4 = v.view(B, T, nh, hd).transpose(1, 2)
    sc = 1.0 / math.sqrt(hd)
    a = (q4 @ k4.transpose(-2, -1)) * sc
    a[:, :, qpos, :] = -1e9
    a[:, :, qpos, wpos] = 20.0
    cm = torch.triu(torch.ones(T, T, device=DEVICE, dtype=torch.bool), diagonal=1)
    a = a.masked_fill(cm.unsqueeze(0).unsqueeze(0), float('-inf'))
    a = F.softmax(a, dim=-1)
    y2 = (a @ v4).transpose(1, 2).contiguous().view(B, T, ne)
    forced_attn1 = block1.attn.c_proj(y2)

    x_after = x_mod + forced_attn1
    mlp_out = block1.mlp(block1.ln_2(x_after))
    final = model.transformer.ln_f(x_after + mlp_out)
    logits = model.lm_head(final)
    return logits[0, qpos].argmax().item()


@torch.no_grad()
def attn_hijack(idx, layer, qpos, wpos):
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


def collect_data():
    """Collect hijack results for all (i, offset) combinations."""
    # results[hijack_type][i_val][offset] = list of bools
    results = {ht: {i: {off: [] for off in OFFSETS} for i in I_VALUES}
               for ht in ['mlp1', 'attn1', 'attn2', 'both']}

    i_set = set(I_VALUES)
    needed = {i: {off: MIN_SAMPLES for off in OFFSETS} for i in I_VALUES}
    t0 = time.time()
    last_progress_batch = 0
    prev_total = 0
    STALL_LIMIT = 5000

    use_targeted = ARGS.offsets is not None and GAP >= 20
    for batch_num in range(MAX_BATCHES):
        if all(needed[i][off] <= 0 for i in I_VALUES for off in OFFSETS):
            break
        if batch_num - last_progress_batch > STALL_LIMIT:
            print(f"  Stall detected at batch {batch_num}, stopping early.", flush=True)
            break

        if use_targeted:
            cval_pick = I_VALUES[batch_num % len(I_VALUES)]
            idx = generate_targeted_batch(cval_pick)
            if idx is None:
                continue
        elif USE_RANDOM:
            idx = generate_random_batch()
        else:
            idx = generate_gap_batch()
            if idx is None:
                continue
        sorted_vals = idx[0, block_size + 1:].cpu().numpy()
        unsorted = idx[0, :block_size].cpu().numpy()
        val_to_pos = {int(unsorted[p]): p for p in range(block_size)}

        target_positions = []
        for p in range(block_size - 1):
            cval = int(sorted_vals[p])
            if cval in i_set:
                tval = int(sorted_vals[p + 1])
                actual_gap = tval - cval
                if actual_gap == GAP:
                    target_positions.append((p, cval, tval))

        if not target_positions:
            continue

        pre = precompute_block0(idx)

        for p, cval, tval in target_positions:
            qpos = block_size + 1 + p

            for off in OFFSETS:
                if needed[cval][off] <= 0:
                    continue
                wval = cval + off
                if wval < 0 or wval == tval or wval >= vocab_n or wval not in val_to_pos:
                    continue
                wpos = val_to_pos[wval]

                pred_mlp1 = mlp1_hijack(pre, idx, qpos, wpos)
                results['mlp1'][cval][off].append(pred_mlp1 == wval)

                pred_attn1 = attn_hijack(idx, 0, qpos, wpos)
                results['attn1'][cval][off].append(pred_attn1 == wval)

                pred_attn2 = attn_hijack(idx, 1, qpos, wpos)
                results['attn2'][cval][off].append(pred_attn2 == wval)

                pred_both = mlp1_and_attn2_hijack(pre, idx, qpos, wpos)
                results['both'][cval][off].append(pred_both == wval)

                needed[cval][off] -= 1

        if (batch_num + 1) % 500 == 0:
            total_done = sum(len(results['mlp1'][i][off]) for i in I_VALUES for off in OFFSETS)
            total_need = len(I_VALUES) * len(OFFSETS) * MIN_SAMPLES
            if total_done > prev_total:
                last_progress_batch = batch_num
                prev_total = total_done
            el = time.time() - t0
            print(f"  batch {batch_num+1}: {total_done}/{total_need} samples "
                  f"({el:.0f}s elapsed)", flush=True)

    return results


def _compute_row_data(results, i_vals_for_row):
    """Compute per-offset rates for a set of i values (pooled)."""
    row_data = {}
    for ht in ['mlp1', 'attn2', 'both', 'and']:
        rates, valid_offs = [], []
        for off in OFFSETS:
            if ht == 'and':
                all_m, all_a = [], []
                for iv in i_vals_for_row:
                    m = results['mlp1'][iv][off]
                    a = results['attn2'][iv][off]
                    n = min(len(m), len(a))
                    all_m.extend(m[:n])
                    all_a.extend(a[:n])
                if len(all_m) >= 5:
                    both_ok = [all_m[j] and all_a[j] for j in range(len(all_m))]
                    rates.append(100 * np.mean(both_ok))
                    valid_offs.append(off)
            else:
                pooled = []
                for iv in i_vals_for_row:
                    pooled.extend(results[ht][iv][off])
                if len(pooled) >= 5:
                    rates.append(100 * np.mean(pooled))
                    valid_offs.append(off)
        row_data[ht] = (valid_offs, rates)
    n_total = sum(len(results['mlp1'][iv][off])
                  for iv in i_vals_for_row for off in OFFSETS)
    return row_data, n_total


def plot_results(results):
    if GROUPS:
        row_items = [(f'$i \\in {{{",".join(map(str,g))}}}$', g) for g in GROUPS]
    else:
        row_items = [(f'$i = {iv}$', [iv]) for iv in I_VALUES]

    n_rows = len(row_items)
    fig, axes = plt.subplots(n_rows, 1, figsize=(7, 2.8 * n_rows), sharey=True)
    if n_rows == 1:
        axes = [axes]

    ht_styles = {
        'mlp1':  {'color': '#d6604d', 'label': 'MLP1 hijack',              'ls': '-'},
        'attn2': {'color': '#2166ac', 'label': 'ATTN2 hijack',             'ls': '-'},
        'both':  {'color': '#4daf4a', 'label': 'Both simultaneously',      'ls': '--'},
        'and':   {'color': '#984ea3', 'label': 'Both individually succeed', 'ls': '--'},
    }

    for row, (label, i_vals) in enumerate(row_items):
        ax = axes[row]
        row_data, n_total = _compute_row_data(results, i_vals)

        for ht, style in ht_styles.items():
            valid_offs, rates = row_data[ht]
            if valid_offs:
                marker = 'o' if style['ls'] == '-' else 's'
                ax.plot(valid_offs, rates, linestyle=style['ls'], marker=marker,
                        color=style['color'], linewidth=1.8, markersize=4,
                        alpha=0.85, label=style['label'] if row == 0 else None)

        ax.set_ylim(-5, 105)
        ax.set_xlim(min(OFFSETS) - GAP * 0.5, max(OFFSETS) + GAP * 0.5)
        ax.set_ylabel(label, fontsize=11, fontweight='bold')
        ax.text(0.98, 0.92, f'n={n_total}', transform=ax.transAxes,
                fontsize=8, ha='right', va='top', color='gray')

        if row == n_rows - 1:
            ax.set_xlabel('Offset (hijack to $i$ + offset)', fontsize=11)

    axes[0].legend(fontsize=9, loc='lower right', framealpha=0.9)
    title_suffix = ' (grouped avg)' if GROUPS else ''
    fig.suptitle(f'Hijack success rate by current value $i$ (gap = {GAP}){title_suffix}',
                 fontsize=14, fontweight='bold', y=1.005)
    fig.tight_layout()
    i_lo, i_hi = I_VALUES[0], I_VALUES[-1]
    fine_tag = '_custom' if ARGS.offsets else ('_fine' if ARGS.fine_offsets else '')
    group_tag = '_grouped' if GROUPS else ''
    extra_tag = f'_{ARGS.out_tag}' if ARGS.out_tag else ''
    out_path = os.path.join(OUTDIR, f'hijack_per_i{i_lo}-{i_hi}_gap{GAP}{fine_tag}{group_tag}{extra_tag}.png')
    fig.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved {out_path}")


if __name__ == '__main__':
    print("Collecting data ...", flush=True)
    results = collect_data()

    for i_val in I_VALUES:
        counts = [len(results['mlp1'][i_val][off]) for off in OFFSETS]
        print(f"  i={i_val}: samples per offset = {counts}")

    plot_results(results)

    if ARGS.save_data:
        import json
        if GROUPS:
            all_i = sorted({i for g in GROUPS for i in g})
        else:
            all_i = I_VALUES
        row_data, n_total = _compute_row_data(results, all_i)
        out = {'offsets': OFFSETS, 'gap': GAP, 'n_total': n_total}
        for ht in ['mlp1', 'attn2', 'both', 'and']:
            v_offs, v_rates = row_data[ht]
            out[ht] = {'offsets': v_offs, 'rates': v_rates}
        os.makedirs(os.path.dirname(ARGS.save_data) or '.', exist_ok=True)
        with open(ARGS.save_data, 'w') as f:
            json.dump(out, f, indent=2)
        print(f"Saved data to {ARGS.save_data}")

    print("Done.", flush=True)
