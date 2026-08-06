#!/usr/bin/env python3
"""Classify k32_N512 seeds as leap-former (two-stage) or single-stage.

Seeds that come out SINGLE are the ones dropped from the cross-seed hijack
average; use --save-json so plot_hijack_avg_seeds.py --classification can read
the verdict instead of relying on a hardcoded seed list.

The verdict is full-sequence accuracy with attn2 ablated: a single-stage model
still sorts perfectly without attn2, a leap-former collapses. Per-token accuracy
is also recorded but is *not* a usable criterion -- ablating attn2 leaves the
first layer able to get 12-71% of individual tokens right across the population,
so the two classes do not separate on it (seeds 8 and 10 score 0.9997 and 1.0000
while the leap-formers spread over 0.12-0.71, with no clean gap near any low
threshold). On full sequences the same models split cleanly at ~0 versus ~1.
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'sortgpt_toolkit'))

import torch
import torch.nn.functional as F
from model import DEVICE, load_model_from_checkpoint

parser = argparse.ArgumentParser()
parser.add_argument('--seeds', default='1-25', help='e.g. "1-25" or "6-25"')
parser.add_argument('--save-json', default=None, help='Write verdicts to this path')
ARGS = parser.parse_args()

_lo, _hi = (int(x) for x in ARGS.seeds.split('-'))
SEED_RANGE = range(_lo, _hi + 1)

N_TRIALS = 200
SINGLE_STAGE_THRESHOLD = 0.5  # on full-sequence accuracy with attn2 ablated

@torch.no_grad()
def classify(model):
    bs = model.config.block_size
    vn = model.config.vocab_size - 1
    b0, b1 = model.transformer.h[0], model.transformer.h[1]
    normal_correct = ablated_correct = total = 0
    normal_fullseq = ablated_fullseq = 0
    for _ in range(N_TRIALS):
        x = torch.randperm(vn)[:bs]
        vals, _ = torch.sort(x)
        sep = torch.tensor([vn])
        idx = torch.cat((x, sep, vals)).unsqueeze(0).to(DEVICE)
        B, T = idx.size()
        targets = idx[0, bs + 1:]
        logits_n, _ = model(idx, block_size=bs, return_full_logits=True)
        preds_n = logits_n[0, bs:2*bs].argmax(dim=-1)
        normal_correct += (preds_n == targets).sum().item()
        normal_fullseq += bool((preds_n == targets).all())
        pos = model.transformer.wpe(model.pos_idx[:T])
        embed = model.transformer.wte(idx) + pos
        x_out = b0(embed)
        x_no_a2 = x_out + b1.mlp(b1.ln_2(x_out)) if b1.mlp is not None else x_out
        x_no_a2 = model.transformer.ln_f(x_no_a2)
        logits_a = x_no_a2 @ model.lm_head.weight.T
        preds_a = logits_a[0, bs:2*bs].argmax(dim=-1)
        ablated_correct += (preds_a == targets).sum().item()
        ablated_fullseq += bool((preds_a == targets).all())
        total += bs
    return {
        'normal_acc': normal_correct / total,
        'ablated_acc': ablated_correct / total,
        'normal_fullseq': normal_fullseq / N_TRIALS,
        'ablated_fullseq': ablated_fullseq / N_TRIALS,
    }

BASE = os.path.join(os.path.dirname(__file__), '..', '..')


def checkpoint_for(seed):
    if seed == 1:
        rel = 'new-grid/k32_N512/checkpoints'
    elif seed <= 5:
        rel = f'new-grid-multiple/k32_N512/seed{seed}/checkpoints'
    elif seed <= 15:
        rel = f'new-grid-multiple-2/k32_N512/seed{seed}/checkpoints'
    else:
        rel = f'new-grid-multiple-3/k32_N512/seed{seed}/checkpoints'
    return os.path.join(BASE, rel, f'std0p01_iseed{seed}__ckpt100000.pt')


results = []
for seed in SEED_RANGE:
    ckpt = checkpoint_for(seed)
    if not os.path.exists(ckpt):
        print(f"seed{seed}: MISSING {ckpt}")
        continue
    model = load_model_from_checkpoint(ckpt)
    m = classify(model)
    m['is_leap'] = m['ablated_fullseq'] <= SINGLE_STAGE_THRESHOLD
    tag = "LEAP" if m['is_leap'] else "SINGLE"
    print(f"seed{seed}: normal={m['normal_acc']:.4f}  no_attn2={m['ablated_acc']:.4f}  "
          f"no_attn2_fullseq={m['ablated_fullseq']:.4f}  -> {tag}")
    results.append((seed, m))

print("\n=== Summary ===")
leaps = [s for s, m in results if m['is_leap']]
singles = [s for s, m in results if not m['is_leap']]
print(f"Leap-formers ({len(leaps)}): seeds {leaps}")
print(f"Single-stage ({len(singles)}): seeds {singles}")

if ARGS.save_json:
    payload = {str(s): m for s, m in results}
    os.makedirs(os.path.dirname(ARGS.save_json) or '.', exist_ok=True)
    with open(ARGS.save_json, 'w') as f:
        json.dump({'criterion': f'ablated_fullseq <= {SINGLE_STAGE_THRESHOLD}',
                   'n_trials': N_TRIALS, 'seeds': payload}, f, indent=2)
    print(f"Saved {ARGS.save_json}")
