#!/usr/bin/env python3
"""
Comprehensive intervention worker for 100k checkpoints (Layer 0).
For each checkpoint: generates N_SEQ random sequences, intervenes at every
sorted-output position with multiple intensities, records per-trial details.

Methodology matches existing perlocation/pernumber experiments:
  - unsorted_lb_num=0, unsorted_ub_num=1 (boost one wrong unsorted number)
  - ub=60 (wide neighbourhood)
  - Same GPTIntervention mechanism from grid-run/model_analysis.py
"""
import argparse
import json
import os
import sys
import time
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'grid-run'))
from model_analysis import GPT, GPTConfig, GPTIntervention

N_SEQ = 3000
INTENSITIES = [2.0, 4.0, 6.0, 10.0]
UB = 60


def remap_state_dict(sd):
    new = {}
    for k, v in sd.items():
        nk = k
        for i in range(10):
            nk = nk.replace(f'transformer.h.{i}.attn.', f'transformer.h.{i}.c_attn.')
            nk = nk.replace(f'transformer.h.{i}.mlp.', f'transformer.h.{i}.c_fc.')
        new[nk] = v
    return new


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    mc = ckpt['model_config']
    config = GPTConfig(block_size=mc['block_size'], vocab_size=mc['vocab_size'] - 1,
                       with_layer_norm=mc.get('use_final_LN', True))
    model = GPT(config)
    sd = remap_state_dict(ckpt['model_state_dict'])
    wpe_max = config.block_size * 4 + 1
    if 'transformer.wpe.weight' in sd and sd['transformer.wpe.weight'].shape[0] > wpe_max:
        sd['transformer.wpe.weight'] = sd['transformer.wpe.weight'][:wpe_max]
    for k in [k for k in sd if k.endswith('.c_attn.bias') and 'c_attn.c_attn' not in k]:
        del sd[k]
    if 'lm_head.weight' in sd:
        del sd['lm_head.weight']
    model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()
    return model, config


def get_batch(vs, bs, device):
    x = torch.randperm(vs)[:bs]
    vals, _ = torch.sort(x)
    return torch.cat((x, torch.tensor([vs]), vals), dim=0).unsqueeze(0).to(device)


@torch.no_grad()
def run_checkpoint(model, config, device):
    bs = config.block_size
    vs = config.vocab_size

    pos_l, num_l, nxt_l, gap_l = [], [], [], []
    int_l, cor_l, pred_l = [], [], []
    n_ok = n_fail = 0

    for si in range(N_SEQ):
        idx = get_batch(vs, bs, device)

        # --- baseline predictions (intensity=0) ---
        logits, _ = model(idx)
        bpreds = torch.argmax(logits, dim=-1)[0]
        for p in range(bs - 1):
            loc = bs + 1 + p
            num_val = idx[0, loc].item()
            nxt_val = idx[0, loc + 1].item()
            pos_l.append(p)
            num_l.append(num_val)
            nxt_l.append(nxt_val)
            gap_l.append(nxt_val - num_val)
            int_l.append(0.0)
            pr = bpreds[loc].item()
            cor_l.append(int(pr == nxt_val))
            pred_l.append(pr)

        # --- interventions ---
        try:
            im = GPTIntervention(model, idx)
        except Exception:
            continue

        for p in range(bs - 1):
            loc = bs + 1 + p
            num_val = idx[0, loc].item()
            nxt_val = idx[0, loc + 1].item()
            gap = nxt_val - num_val
            for intensity in INTENSITIES:
                try:
                    im.intervent_attention(
                        attention_layer_num=0, location=loc,
                        unsorted_lb=UB, unsorted_ub=UB,
                        unsorted_lb_num=0, unsorted_ub_num=1,
                        unsorted_intensity_inc=intensity,
                        sorted_lb=0, sorted_num=0, sorted_intensity_inc=0.0)
                    pr, ac = im.check_if_still_works()
                    pos_l.append(p)
                    num_l.append(num_val)
                    nxt_l.append(nxt_val)
                    gap_l.append(gap)
                    int_l.append(intensity)
                    cor_l.append(int(pr == ac))
                    pred_l.append(pr)
                    im.revert_attention(0)
                    n_ok += 1
                except Exception:
                    try:
                        im.revert_attention(0)
                    except Exception:
                        pass
                    n_fail += 1

        if (si + 1) % 500 == 0:
            print(f"    {si+1}/{N_SEQ}  ok={n_ok} fail={n_fail}", flush=True)

    return dict(
        position=np.array(pos_l, dtype=np.int16),
        number=np.array(num_l, dtype=np.int16),
        next_number=np.array(nxt_l, dtype=np.int16),
        gap=np.array(gap_l, dtype=np.int16),
        intensity=np.array(int_l, dtype=np.float32),
        correct=np.array(cor_l, dtype=np.int8),
        predicted=np.array(pred_l, dtype=np.int16),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tasks-file', required=True)
    ap.add_argument('--gpu', type=int, required=True)
    args = ap.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = 'cuda'

    with open(args.tasks_file) as f:
        tasks = json.load(f)
    print(f"GPU {args.gpu}: {len(tasks)} checkpoints to process", flush=True)

    for t in tasks:
        if os.path.exists(t['out']):
            print(f"  Skip {t['name']} (cached)", flush=True)
            continue
        t0 = time.time()
        model, config = load_model(t['ckpt_path'], device)
        print(f"  Loaded {t['name']} ({time.time()-t0:.1f}s)", flush=True)

        t0 = time.time()
        res = run_checkpoint(model, config, device)
        os.makedirs(os.path.dirname(t['out']), exist_ok=True)
        np.savez_compressed(t['out'], **res)
        dt = time.time() - t0
        n = len(res['position'])
        print(json.dumps({
            'done': t['name'], 'gpu': args.gpu,
            'elapsed': round(dt, 1), 'n_trials': n
        }), flush=True)

    print(f"GPU {args.gpu}: all done", flush=True)


if __name__ == '__main__':
    main()
