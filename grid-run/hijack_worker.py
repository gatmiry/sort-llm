"""
GPU worker for hijack intervention on grid-run checkpoints.
Handles grid-run checkpoint format: ckpt['model'], ckpt['config'].
"""
import argparse
import json
import os
import sys
import time
import types
import numpy as np
import torch
import torch.nn.functional as F
from model_analysis import GPT, GPTConfig

INTENSITY = 10.0


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    cfg = ckpt['config']
    config = GPTConfig(block_size=cfg['block_size'], vocab_size=cfg['vocab_size'],
                       with_layer_norm=cfg['with_layer_norm'])
    model = GPT(config)
    model.load_state_dict(ckpt['model'])
    model.to(device)
    model.eval()
    return model, config


def get_batch(vs, bs, device):
    x = torch.randperm(vs)[:bs]
    vals, _ = torch.sort(x)
    return torch.cat((x, torch.tensor([vs]), vals), dim=0).unsqueeze(0).to(device)


def run_hijack(model, config, device, n_trials=2000):
    bs = config.block_size
    vs = config.vocab_size
    attn_module = model.transformer.h[0].c_attn
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

    return np.array(records, dtype=np.int32) if records else np.empty((0, 4), dtype=np.int32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks-file', required=True)
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = 'cuda'

    with open(args.tasks_file) as f:
        task_list = json.load(f)

    print(f"GPU {args.gpu}: {len(task_list)} tasks", flush=True)

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
            data = run_hijack(model, config, device, n_trials=task.get('trials', 2000))
            np.savez(task['out'], data=data, vocab_size=config.vocab_size)
            dt = time.time() - t0
            done += 1
            print(json.dumps({
                'status': 'done', 'task': task['name'], 'gpu': args.gpu,
                'elapsed': round(dt, 1), 'progress': f'{done}/{len(task_list)}',
                'records': len(data),
            }), flush=True)
        except Exception as e:
            done += 1
            print(json.dumps({
                'status': 'fail', 'task': task['name'], 'error': str(e),
            }), flush=True)

    print(f"GPU {args.gpu}: all done ({done}/{len(task_list)})", flush=True)


if __name__ == '__main__':
    main()
