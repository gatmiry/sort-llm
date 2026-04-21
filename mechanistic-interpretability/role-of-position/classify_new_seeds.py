#!/usr/bin/env python3
"""Classify seeds 6-25 for k32_N512 as leap-former or not."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'sortgpt_toolkit'))

import torch
import torch.nn.functional as F
from model import DEVICE, load_model_from_checkpoint

N_TRIALS = 200

@torch.no_grad()
def classify(model):
    bs = model.config.block_size
    vn = model.config.vocab_size - 1
    b0, b1 = model.transformer.h[0], model.transformer.h[1]
    normal_correct = ablated_correct = total = 0
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
        pos = model.transformer.wpe(model.pos_idx[:T])
        embed = model.transformer.wte(idx) + pos
        x_out = b0(embed)
        x_no_a2 = x_out + b1.mlp(b1.ln_2(x_out)) if b1.mlp is not None else x_out
        x_no_a2 = model.transformer.ln_f(x_no_a2)
        logits_a = x_no_a2 @ model.lm_head.weight.T
        preds_a = logits_a[0, bs:2*bs].argmax(dim=-1)
        ablated_correct += (preds_a == targets).sum().item()
        total += bs
    return normal_correct / total, ablated_correct / total

BASE = os.path.join(os.path.dirname(__file__), '..', '..')
results = []
for seed in range(6, 26):
    if seed <= 15:
        ckpt = os.path.join(BASE, f'new-grid-multiple-2/k32_N512/seed{seed}/checkpoints/std0p01_iseed{seed}__ckpt100000.pt')
    else:
        ckpt = os.path.join(BASE, f'new-grid-multiple-3/k32_N512/seed{seed}/checkpoints/std0p01_iseed{seed}__ckpt100000.pt')
    model = load_model_from_checkpoint(ckpt)
    norm_acc, abl_acc = classify(model)
    is_leap = abl_acc < 0.10
    tag = "LEAP" if is_leap else "SINGLE"
    print(f"seed{seed}: normal={norm_acc:.4f}  no_attn2={abl_acc:.4f}  -> {tag}")
    results.append((seed, norm_acc, abl_acc, is_leap))

print("\n=== Summary ===")
leaps = [s for s, _, _, lf in results if lf]
singles = [s for s, _, _, lf in results if not lf]
print(f"Leap-formers ({len(leaps)}): seeds {leaps}")
print(f"Single-stage ({len(singles)}): seeds {singles}")
