"""
GPU worker for per-number intervention experiments.
For each target number, generates sequences containing that number,
finds its position in the sorted output, and intervenes there.
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

INTENSITY_VALUES = [1.0, 2.0, 4.0, 6.0, 8.0, 10.0]
UB = 60
LB = 60
MIN_VALID = 200


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


def get_batch_with_number(vs, bs, target_num, device):
    """Generate a batch that always contains target_num."""
    pool = list(range(vs))
    pool.remove(target_num)
    perm = torch.randperm(len(pool))[:bs - 1]
    others = torch.tensor([pool[i] for i in perm])
    x = torch.cat([others, torch.tensor([target_num])])
    x = x[torch.randperm(bs)]
    vals, _ = torch.sort(x)
    return torch.cat((x, torch.tensor([vs]), vals), dim=0).unsqueeze(0).to(device)


def find_location_of_number(idx, bs, target_num):
    """Find the location index for target_num in the sorted output.
    Returns None if target_num is the last sorted element (no next token to predict)."""
    sorted_part = idx[0, bs + 1: 2 * bs + 1]
    matches = (sorted_part == target_num).nonzero(as_tuple=True)[0]
    if len(matches) == 0:
        return None
    k = matches[0].item()
    if k >= bs - 1:
        return None
    return bs + 1 + k


def compute_intensity_for_number(model, config, device, attn_layer, target_num):
    bs = config.block_size
    vs = config.vocab_size

    rates, counts = [], []
    for intens in INTENSITY_VALUES:
        attempts, rounds = [], 0
        while len(attempts) < MIN_VALID and rounds < 3000:
            rounds += 1
            idx = get_batch_with_number(vs, bs, target_num, device)
            location = find_location_of_number(idx, bs, target_num)
            if location is None:
                continue
            try:
                im = GPTIntervention(model, idx)
                im.intervent_attention(
                    attention_layer_num=attn_layer, location=location,
                    unsorted_lb=LB, unsorted_ub=UB,
                    unsorted_lb_num=0, unsorted_ub_num=1,
                    unsorted_intensity_inc=intens,
                    sorted_lb=0, sorted_num=0, sorted_intensity_inc=0.0)
                g, n = im.check_if_still_works()
                attempts.append(g == n)
                im.revert_attention(attn_layer)
            except:
                continue
        counts.append(len(attempts))
        rates.append(sum(attempts) / len(attempts) if attempts else 0.0)

    return np.array(INTENSITY_VALUES), np.array(rates), np.array(counts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks-file', required=True)
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = 'cuda'

    with open(args.tasks_file) as f:
        task_list = json.load(f)

    n_ckpts = len(set(t['ckpt_path'] for t in task_list))
    print(f"GPU {args.gpu}: {len(task_list)} tasks across {n_ckpts} checkpoints", flush=True)

    current_ckpt = None
    model = None
    done = 0

    for task in task_list:
        if os.path.exists(task['out']):
            done += 1
            continue

        if task['ckpt_path'] != current_ckpt:
            t0 = time.time()
            model, config = load_model(task['ckpt_path'], device)
            current_ckpt = task['ckpt_path']
            print(f"  Loaded {os.path.basename(current_ckpt)} ({time.time()-t0:.1f}s)", flush=True)

        os.makedirs(os.path.dirname(task['out']), exist_ok=True)
        t0 = time.time()
        try:
            intensities, rates, counts = compute_intensity_for_number(
                model, config, device, task['layer'], task['target_num'])
            np.savez(task['out'], intensities=intensities, success_rates=rates,
                     counts=counts, target_num=task['target_num'], layer=task['layer'])
            dt = time.time() - t0
            done += 1
            print(json.dumps({
                'status': 'done', 'task': task['name'], 'gpu': args.gpu,
                'elapsed': round(dt, 1), 'progress': f'{done}/{len(task_list)}',
                'counts': counts.tolist(),
            }), flush=True)
        except Exception as e:
            done += 1
            print(json.dumps({
                'status': 'fail', 'task': task['name'], 'error': str(e),
            }), flush=True)

    print(f"GPU {args.gpu}: all done ({done}/{len(task_list)})", flush=True)


if __name__ == '__main__':
    main()
