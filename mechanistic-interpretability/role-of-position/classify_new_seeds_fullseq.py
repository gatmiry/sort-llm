#!/usr/bin/env python3
"""Classify seeds 6-25 for k32_N512 using full-sequence accuracy."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'sortgpt_toolkit'))

import torch
from model import DEVICE, load_model_from_checkpoint

N_TRIALS = 500

@torch.no_grad()
def classify(model):
    bs = model.config.block_size
    vn = model.config.vocab_size - 1
    b0, b1 = model.transformer.h[0], model.transformer.h[1]
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
        if (preds_n == targets).all():
            normal_fullseq += 1

        pos = model.transformer.wpe(model.pos_idx[:T])
        embed = model.transformer.wte(idx) + pos
        x_out = b0(embed)
        x_no_a2 = x_out + b1.mlp(b1.ln_2(x_out)) if b1.mlp is not None else x_out
        x_no_a2 = model.transformer.ln_f(x_no_a2)
        logits_a = x_no_a2 @ model.lm_head.weight.T
        preds_a = logits_a[0, bs:2*bs].argmax(dim=-1)
        if (preds_a == targets).all():
            ablated_fullseq += 1

    return normal_fullseq / N_TRIALS, ablated_fullseq / N_TRIALS

BASE = os.path.join(os.path.dirname(__file__), '..', '..')
print(f"{'seed':<8} {'normal':>8} {'no_attn2':>10}  type")
print("-" * 40)
for seed in range(6, 26):
    if seed <= 15:
        ckpt = os.path.join(BASE, f'new-grid-multiple-2/k32_N512/seed{seed}/checkpoints/std0p01_iseed{seed}__ckpt100000.pt')
    else:
        ckpt = os.path.join(BASE, f'new-grid-multiple-3/k32_N512/seed{seed}/checkpoints/std0p01_iseed{seed}__ckpt100000.pt')
    model = load_model_from_checkpoint(ckpt)
    norm, abl = classify(model)
    tag = "SINGLE" if abl > 0.5 else "TWO-STAGE"
    print(f"seed{seed:<4} {norm:>8.3f} {abl:>10.3f}  {tag}")
