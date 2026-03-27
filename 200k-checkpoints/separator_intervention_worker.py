"""
GPU worker for separator-attention and random-target intervention experiments.
For each random sequence, at each sorted output position:
  1. Check if layer 0 max attention is on the separator token.
     If yes, intervene with standard method (ub=60) and record result.
  2. Intervene by boosting a random unsorted number's attention and record result.
Collects per-number success data across many trials.
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

INTENSITIES = [2.0, 6.0, 10.0]
UB_STANDARD = 60


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


def get_batch(vocab_size, block_size, device):
    x = torch.randperm(vocab_size)[:block_size]
    vals, _ = torch.sort(x)
    return torch.cat((x, torch.tensor([vocab_size]), vals), dim=0).unsqueeze(0).to(device)


def try_standard_intervention(model, idx, config, location, intensity):
    try:
        im = GPTIntervention(model, idx)
        im.intervent_attention(
            attention_layer_num=0, location=location,
            unsorted_lb=UB_STANDARD, unsorted_ub=UB_STANDARD,
            unsorted_lb_num=0, unsorted_ub_num=1,
            unsorted_intensity_inc=intensity,
            sorted_lb=0, sorted_num=0, sorted_intensity_inc=0.0)
        g, n = im.check_if_still_works()
        im.revert_attention(0)
        return g == n
    except:
        return None


def try_random_intervention(model, idx, config, location, intensity):
    vs = config.vocab_size
    try:
        im = GPTIntervention(model, idx)
        im.intervent_attention(
            attention_layer_num=0, location=location,
            unsorted_lb=0, unsorted_ub=vs,
            unsorted_lb_num=0, unsorted_ub_num=1,
            unsorted_intensity_inc=intensity,
            sorted_lb=0, sorted_num=0, sorted_intensity_inc=0.0)
        g, n = im.check_if_still_works()
        im.revert_attention(0)
        return g == n
    except:
        pass
    try:
        im = GPTIntervention(model, idx)
        im.intervent_attention(
            attention_layer_num=0, location=location,
            unsorted_lb=vs, unsorted_ub=0,
            unsorted_lb_num=1, unsorted_ub_num=0,
            unsorted_intensity_inc=intensity,
            sorted_lb=0, sorted_num=0, sorted_intensity_inc=0.0)
        g, n = im.check_if_still_works()
        im.revert_attention(0)
        return g == n
    except:
        return None


def run_trials(model, config, device, n_trials):
    bs = config.block_size
    vs = config.vocab_size
    sep_pos = bs

    sep_records = []
    rand_records = []

    for trial in range(n_trials):
        idx = get_batch(vs, bs, device)

        with torch.no_grad():
            logits, _ = model(idx)

        attn_layer0 = model.transformer.h[0].c_attn.attn

        for p in range(bs - 1):
            sorted_loc = bs + 1 + p
            number_val = idx[0, sorted_loc].item()
            next_num = idx[0, sorted_loc + 1].item()

            attn_row = attn_layer0[sorted_loc, :sorted_loc + 1]
            max_attn_pos = attn_row.argmax().item()
            attends_to_sep = (max_attn_pos == sep_pos)

            for intensity in INTENSITIES:
                if attends_to_sep:
                    result = try_standard_intervention(model, idx, config, sorted_loc, intensity)
                    if result is not None:
                        sep_records.append((number_val, intensity, int(result)))

                result_rand = try_random_intervention(model, idx, config, sorted_loc, intensity)
                if result_rand is not None:
                    rand_records.append((number_val, intensity, int(result_rand)))

        if (trial + 1) % 200 == 0:
            print(f"  Trial {trial+1}/{n_trials}: sep={len(sep_records)}, rand={len(rand_records)}",
                  flush=True)

    return np.array(sep_records, dtype=np.int32), np.array(rand_records, dtype=np.int32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--trials', type=int, default=1000)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = 'cuda'

    print(f"GPU {args.gpu}: loading model...", flush=True)
    t0 = time.time()
    model, config = load_model(args.ckpt, device)
    print(f"  Loaded in {time.time()-t0:.1f}s", flush=True)

    print(f"GPU {args.gpu}: running {args.trials} trials...", flush=True)
    t0 = time.time()
    sep_data, rand_data = run_trials(model, config, device, args.trials)
    elapsed = time.time() - t0

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, sep_data=sep_data, rand_data=rand_data)
    print(f"GPU {args.gpu}: done in {elapsed:.0f}s, "
          f"sep={len(sep_data)} rand={len(rand_data)}", flush=True)


if __name__ == '__main__':
    main()
