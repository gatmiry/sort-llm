#!/usr/bin/env python3
"""
Test two residual stream removal claims across all leap-former checkpoints:

Claim 1: attn1_out can be removed from the residual stream AFTER it has fed
          mlp1 (i.e., remove from everything downstream: attn2, mlp2, ln_f+lm_head)
          without changing argmax predictions.

Claim 2: mlp1_out can be removed from the residual stream BEFORE ln_f + lm_head
          (i.e., keep it for attn2 and mlp2 inputs, but subtract from final readout)
          without changing argmax predictions.

Steps:
  1. Classify all 40 checkpoints as leap-former or base-former (attn2 ablation).
  2. For leap-formers, test both claims.
"""
import os, sys, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'sortgpt_toolkit'))

import numpy as np
import torch
import torch.nn.functional as F

from model import DEVICE, load_model_from_checkpoint, GPT

GRID_DIR = os.path.join(os.path.dirname(__file__), '..', 'new-grid')
MULTI_DIR = os.path.join(os.path.dirname(__file__), '..', 'new-grid-multiple')
OUTFILE = os.path.join(os.path.dirname(__file__), 'residual_removal_results.json')

N_CLASSIFY = 200
N_TEST = 500


def find_all_checkpoints():
    """Collect final (ckpt100000) checkpoint paths for all 40 models."""
    ckpts = []
    for cfg in ['k16_N128', 'k16_N256', 'k16_N512', 'k16_N1024',
                'k32_N128', 'k32_N256', 'k32_N512', 'k32_N1024']:
        # seed 1 in new-grid
        p = os.path.join(GRID_DIR, cfg, 'checkpoints')
        for f in sorted(os.listdir(p)):
            if f.endswith('_ckpt100000.pt') and 'iseed1' in f:
                ckpts.append({'path': os.path.join(p, f), 'config': cfg, 'seed': 1})
                break
        # seeds 2-5 in new-grid-multiple
        for seed in [2, 3, 4, 5]:
            p2 = os.path.join(MULTI_DIR, cfg, f'seed{seed}', 'checkpoints')
            if not os.path.isdir(p2):
                continue
            for f in sorted(os.listdir(p2)):
                if f.endswith('_ckpt100000.pt') and f'iseed{seed}' in f:
                    ckpts.append({'path': os.path.join(p2, f), 'config': cfg, 'seed': seed})
                    break
    return ckpts


def get_batch(vocab_n, block_size, device):
    x = torch.randperm(vocab_n)[:block_size]
    vals, _ = torch.sort(x)
    sep = torch.tensor([vocab_n])
    return torch.cat((x, sep, vals)).unsqueeze(0).to(device)


@torch.no_grad()
def classify_leapformer(model, n_trials=N_CLASSIFY):
    """Return (normal_acc, no_attn2_acc). Leap-former if no_attn2 drops to ~0."""
    bs = model.config.block_size
    vn = model.config.vocab_size - 1
    b0, b1 = model.transformer.h[0], model.transformer.h[1]

    normal_correct = 0
    ablated_correct = 0
    total = 0

    for _ in range(n_trials):
        idx = get_batch(vn, bs, DEVICE)
        B, T = idx.size()
        targets = idx[0, bs + 1:]

        # Normal forward
        logits_n, _ = model(idx, block_size=bs, return_full_logits=True)
        preds_n = logits_n[0, bs:2*bs].argmax(dim=-1)
        normal_correct += (preds_n == targets).sum().item()

        # Ablate attn2: skip block1.attn, keep block1.mlp
        pos = model.transformer.wpe(model.pos_idx[:T])
        embed = model.transformer.wte(idx) + pos
        x = b0(embed)
        # skip attn2: x = x + 0 (no attn)
        x_no_a2 = x + b1.mlp(b1.ln_2(x)) if b1.mlp is not None else x
        x_no_a2 = model.transformer.ln_f(x_no_a2)
        logits_a = x_no_a2 @ model.lm_head.weight.T
        preds_a = logits_a[0, bs:2*bs].argmax(dim=-1)
        ablated_correct += (preds_a == targets).sum().item()

        total += bs

    return normal_correct / total, ablated_correct / total


@torch.no_grad()
def test_residual_removal(model, n_trials=N_TEST):
    """
    Test both claims. Returns dict with per-position agreement rates.

    For each trial and each sorted position, check whether argmax prediction
    matches the normal forward pass under two interventions.
    """
    bs = model.config.block_size
    vn = model.config.vocab_size - 1
    n_embd = model.config.n_embd
    b0, b1 = model.transformer.h[0], model.transformer.h[1]

    claim1_agree = 0  # attn1 removal from residual after feeding mlp1
    claim2_agree = 0  # mlp1 removal from final readout
    both_agree = 0
    total = 0

    for _ in range(n_trials):
        idx = get_batch(vn, bs, DEVICE)
        B, T = idx.size()

        # ── Normal forward (manual, to extract components) ──
        pos = model.transformer.wpe(model.pos_idx[:T])
        embed = model.transformer.wte(idx) + pos

        # Block0
        h0 = b0.ln_1(embed)
        qkv0 = b0.attn.c_attn(h0)
        q0, k0, v0 = qkv0.split(n_embd, dim=2)
        q0h = q0.view(B, T, 1, n_embd).transpose(1, 2)
        k0h = k0.view(B, T, 1, n_embd).transpose(1, 2)
        v0h = v0.view(B, T, 1, n_embd).transpose(1, 2)
        y0 = F.scaled_dot_product_attention(q0h, k0h, v0h, dropout_p=0.0, is_causal=True)
        attn1_out = b0.attn.c_proj(y0.transpose(1, 2).contiguous().view(B, T, n_embd))

        res_after_attn1 = embed + attn1_out
        mlp1_out = b0.mlp(b0.ln_2(res_after_attn1))
        res_after_block0 = res_after_attn1 + mlp1_out  # = embed + attn1_out + mlp1_out

        # Block1 normal
        h1 = b1.ln_1(res_after_block0)
        qkv1 = b1.attn.c_attn(h1)
        q1, k1, v1 = qkv1.split(n_embd, dim=2)
        q1h = q1.view(B, T, 1, n_embd).transpose(1, 2)
        k1h = k1.view(B, T, 1, n_embd).transpose(1, 2)
        v1h = v1.view(B, T, 1, n_embd).transpose(1, 2)
        y1 = F.scaled_dot_product_attention(q1h, k1h, v1h, dropout_p=0.0, is_causal=True)
        attn2_out = b1.attn.c_proj(y1.transpose(1, 2).contiguous().view(B, T, n_embd))

        res_after_attn2 = res_after_block0 + attn2_out
        mlp2_out = b1.mlp(b1.ln_2(res_after_attn2))
        res_final = res_after_attn2 + mlp2_out
        # = embed + attn1_out + mlp1_out + attn2_out + mlp2_out

        logits_normal = model.lm_head(model.transformer.ln_f(res_final))
        preds_normal = logits_normal[0, bs:2*bs].argmax(dim=-1)

        # ── Claim 1: Remove attn1_out from residual after it fed mlp1 ──
        # attn1_out was used to compute mlp1 (via res_after_attn1 = embed + attn1_out)
        # Now remove it from everything downstream:
        #   residual entering block1 = embed + mlp1_out  (no attn1_out)
        res_no_a1 = embed + mlp1_out  # attn1_out removed

        h1_c1 = b1.ln_1(res_no_a1)
        qkv1_c1 = b1.attn.c_attn(h1_c1)
        q1c, k1c, v1c = qkv1_c1.split(n_embd, dim=2)
        q1ch = q1c.view(B, T, 1, n_embd).transpose(1, 2)
        k1ch = k1c.view(B, T, 1, n_embd).transpose(1, 2)
        v1ch = v1c.view(B, T, 1, n_embd).transpose(1, 2)
        y1_c1 = F.scaled_dot_product_attention(q1ch, k1ch, v1ch, dropout_p=0.0, is_causal=True)
        attn2_c1 = b1.attn.c_proj(y1_c1.transpose(1, 2).contiguous().view(B, T, n_embd))

        res_after_attn2_c1 = res_no_a1 + attn2_c1
        mlp2_c1 = b1.mlp(b1.ln_2(res_after_attn2_c1))
        res_final_c1 = res_after_attn2_c1 + mlp2_c1

        logits_c1 = model.lm_head(model.transformer.ln_f(res_final_c1))
        preds_c1 = logits_c1[0, bs:2*bs].argmax(dim=-1)

        # ── Claim 2: Remove mlp1_out from final readout only ──
        # mlp1_out stays for attn2 and mlp2 inputs (normal computation)
        # but is subtracted from the final residual before ln_f + lm_head
        res_final_c2 = res_final - mlp1_out
        # = embed + attn1_out + attn2_out + mlp2_out (no mlp1_out at readout)

        logits_c2 = model.lm_head(model.transformer.ln_f(res_final_c2))
        preds_c2 = logits_c2[0, bs:2*bs].argmax(dim=-1)

        # Count agreements
        n = bs  # number of sorted positions
        c1_match = (preds_c1 == preds_normal).sum().item()
        c2_match = (preds_c2 == preds_normal).sum().item()
        both_match = ((preds_c1 == preds_normal) & (preds_c2 == preds_normal)).sum().item()

        claim1_agree += c1_match
        claim2_agree += c2_match
        both_agree += both_match
        total += n

    return {
        'claim1_agree_pct': 100 * claim1_agree / total,
        'claim2_agree_pct': 100 * claim2_agree / total,
        'both_agree_pct': 100 * both_agree / total,
        'total_positions': total,
    }


def main():
    ckpts = find_all_checkpoints()
    print(f"Found {len(ckpts)} checkpoints", flush=True)

    results = []
    leap_count = 0
    t0 = time.time()

    for i, ck in enumerate(ckpts):
        label = f"{ck['config']}_s{ck['seed']}"
        print(f"\n[{i+1}/{len(ckpts)}] {label}", flush=True)

        model = load_model_from_checkpoint(ck['path'])

        # Step 1: classify
        normal_acc, no_attn2_acc = classify_leapformer(model)
        is_leap = no_attn2_acc < 0.90
        tag = "LEAP" if is_leap else "BASE"
        if is_leap:
            leap_count += 1
        print(f"  {tag}: normal={normal_acc:.3f}, no_attn2={no_attn2_acc:.3f}", flush=True)

        entry = {
            'label': label, 'config': ck['config'], 'seed': ck['seed'],
            'normal_acc': round(normal_acc, 4),
            'no_attn2_acc': round(no_attn2_acc, 4),
            'is_leapformer': is_leap,
        }

        # Step 2: test claims on leap-formers
        if is_leap:
            res = test_residual_removal(model)
            entry.update(res)
            print(f"  Claim1 (remove attn1): {res['claim1_agree_pct']:.2f}% agree", flush=True)
            print(f"  Claim2 (remove mlp1 at readout): {res['claim2_agree_pct']:.2f}% agree", flush=True)
            print(f"  Both: {res['both_agree_pct']:.2f}%", flush=True)

        results.append(entry)
        del model
        torch.cuda.empty_cache()

    elapsed = time.time() - t0

    with open(OUTFILE, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {OUTFILE}")

    print(f"\n{'='*70}")
    print(f"SUMMARY  ({elapsed:.0f}s)")
    print(f"{'='*70}")
    print(f"Total checkpoints: {len(ckpts)}")
    print(f"Leap-formers: {leap_count}")

    leaps = [r for r in results if r['is_leapformer']]
    if leaps:
        print(f"\nLeap-former results:")
        print(f"{'Model':<22} {'normal':>7} {'no_a2':>7} {'Claim1':>8} {'Claim2':>8} {'Both':>8}")
        print("-" * 65)
        for r in leaps:
            print(f"{r['label']:<22} {r['normal_acc']:>7.3f} {r['no_attn2_acc']:>7.3f} "
                  f"{r['claim1_agree_pct']:>7.2f}% {r['claim2_agree_pct']:>7.2f}% "
                  f"{r['both_agree_pct']:>7.2f}%")

        c1_vals = [r['claim1_agree_pct'] for r in leaps]
        c2_vals = [r['claim2_agree_pct'] for r in leaps]
        print(f"\nClaim1 range: {min(c1_vals):.2f}% – {max(c1_vals):.2f}%  (mean {np.mean(c1_vals):.2f}%)")
        print(f"Claim2 range: {min(c2_vals):.2f}% – {max(c2_vals):.2f}%  (mean {np.mean(c2_vals):.2f}%)")

    bases = [r for r in results if not r['is_leapformer']]
    if bases:
        print(f"\nBase-former models ({len(bases)}):")
        for r in bases:
            print(f"  {r['label']:<22} normal={r['normal_acc']:.3f} no_attn2={r['no_attn2_acc']:.3f}")


if __name__ == '__main__':
    main()
