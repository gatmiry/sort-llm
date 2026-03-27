"""
GPU worker: processes a batch of analysis tasks on a single GPU.
Model is loaded once per checkpoint and reused for all tasks on that checkpoint.
Prints JSON status lines so the launcher can track progress.
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


def remap_state_dict(sd_100k):
    new_sd = {}
    for key, val in sd_100k.items():
        new_key = key
        for i in range(10):
            new_key = new_key.replace(f'transformer.h.{i}.attn.', f'transformer.h.{i}.c_attn.')
            new_key = new_key.replace(f'transformer.h.{i}.mlp.', f'transformer.h.{i}.c_fc.')
        new_sd[new_key] = val
    return new_sd


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    mc = ckpt['model_config']
    vocab_size = mc['vocab_size'] - 1
    block_size = mc['block_size']
    with_layer_norm = mc.get('use_final_LN', True)

    config = GPTConfig(block_size=block_size, vocab_size=vocab_size,
                       with_layer_norm=with_layer_norm)
    model = GPT(config)

    sd = remap_state_dict(ckpt['model_state_dict'])
    grid_wpe_size = block_size * 4 + 1
    if 'transformer.wpe.weight' in sd and sd['transformer.wpe.weight'].shape[0] > grid_wpe_size:
        sd['transformer.wpe.weight'] = sd['transformer.wpe.weight'][:grid_wpe_size]
    keys_to_skip = [k for k in sd if k.endswith('.c_attn.bias') and 'c_attn.c_attn' not in k]
    for k in keys_to_skip:
        del sd[k]
    if 'lm_head.weight' in sd:
        del sd['lm_head.weight']

    model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()
    return model, config


def get_batch(vocab_size, block_size, device='cpu'):
    x = torch.randperm(vocab_size)[:block_size]
    vals, _ = torch.sort(x)
    return torch.cat((x, torch.tensor([vocab_size]), vals), dim=0).unsqueeze(0).to(device)


def compute_cinclogits(model, config, device, attn_layer, num_tries=100):
    bs = config.block_size
    vs = config.vocab_size
    acc_cl = np.zeros(bs)
    acc_icl = np.zeros(bs)
    for _ in range(num_tries):
        idx = get_batch(vs, bs, device)
        with torch.no_grad():
            logits, _ = model(idx)
        is_correct = (torch.argmax(logits[0, bs:2*bs, :], dim=1) == idx[0, bs+1:])
        attn_w = model.transformer.h[attn_layer].c_attn.attn
        for j in range(bs, 2*bs):
            max_s, max_n = float('-inf'), -1
            for k in range(2*bs+1):
                s = attn_w[j, k].item()
                if s > max_s:
                    max_s = s
                    max_n = idx[0, k].item()
            sc = (max_n == idx[0, j+1].item())
            pos = j - bs
            lc = is_correct[pos].item()
            if lc and not sc:
                acc_cl[pos] += 1.0
            elif not lc and not sc:
                acc_icl[pos] += 1.0
    return acc_cl / num_tries, acc_icl / num_tries


def compute_intensity(model, config, device, attn_layer, ub=5, min_valid=200):
    bs = config.block_size
    vs = config.vocab_size
    location = bs + 5
    intensities = [1.0, 2.0, 4.0, 6.0, 8.0, 10.0]
    rates, counts = [], []
    for intens in intensities:
        attempts, rounds = [], 0
        while len(attempts) < min_valid and rounds < 2000:
            rounds += 1
            idx = get_batch(vs, bs, device)
            try:
                im = GPTIntervention(model, idx)
                im.intervent_attention(
                    attention_layer_num=attn_layer, location=location,
                    unsorted_lb=ub, unsorted_ub=ub,
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
    return np.array(intensities), np.array(rates), np.array(counts)


def compute_ablation(model, config, device, skip_layer, num_trials=500):
    bs = config.block_size
    vs = config.vocab_size
    block = model.transformer.h[skip_layer]
    orig_fwd = block.forward

    def skip_attn(x, layer_n=-1):
        return x + block.c_fc(block.ln_2(x))
    block.forward = skip_attn

    pp = np.zeros(bs)
    fs = 0
    cc = np.zeros(bs)
    ce = np.zeros(bs)
    try:
        for _ in range(num_trials):
            idx = get_batch(vs, bs, device)
            with torch.no_grad():
                logits, _ = model(idx)
            preds = torch.argmax(logits[0, bs:2*bs, :], dim=1)
            targets = idx[0, bs+1:]
            correct = (preds == targets).cpu().numpy()
            pp += correct
            if correct.all():
                fs += 1
            ok = True
            for i in range(bs):
                if ok:
                    ce[i] += 1
                    if correct[i]:
                        cc[i] += 1
                    else:
                        ok = False
                else:
                    break
    finally:
        block.forward = orig_fwd
    return pp / num_trials, fs / num_trials, np.where(ce > 0, cc / ce, 0.0), ce


def compute_baseline(model, config, device, num_trials=500):
    bs = config.block_size
    vs = config.vocab_size
    pp = np.zeros(bs)
    fs = 0
    cc = np.zeros(bs)
    ce = np.zeros(bs)
    for _ in range(num_trials):
        idx = get_batch(vs, bs, device)
        with torch.no_grad():
            logits, _ = model(idx)
        preds = torch.argmax(logits[0, bs:2*bs, :], dim=1)
        targets = idx[0, bs+1:]
        correct = (preds == targets).cpu().numpy()
        pp += correct
        if correct.all():
            fs += 1
        ok = True
        for i in range(bs):
            if ok:
                ce[i] += 1
                if correct[i]:
                    cc[i] += 1
                else:
                    ok = False
            else:
                break
    return pp / num_trials, fs / num_trials, np.where(ce > 0, cc / ce, 0.0), ce


def process_task(task, model, config, device, out_dir, itr):
    task_type = task['type']
    out_path = task['out']
    if os.path.exists(out_path):
        return True

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if task_type == 'baseline':
        pp, fs, ca, ce = compute_baseline(model, config, device)
        np.savez(out_path, per_pos_acc=pp, full_seq_acc=fs,
                 cond_acc=ca, cond_eligible=ce, itr=itr)
    elif task_type == 'ablation':
        pp, fs, ca, ce = compute_ablation(model, config, device, task['layer'])
        np.savez(out_path, per_pos_acc=pp, full_seq_acc=fs,
                 cond_acc=ca, cond_eligible=ce, skip_layer=task['layer'], itr=itr)
    elif task_type == 'cinclogits':
        cl, icl = compute_cinclogits(model, config, device, task['layer'])
        np.savez(out_path, clogit_icscore=cl, iclogit_icscore=icl, itr=itr)
    elif task_type == 'intensity':
        intensities, rates, counts = compute_intensity(
            model, config, device, task['layer'], ub=task['ub'])
        np.savez(out_path, intensities=intensities, success_rates=rates,
                 counts=counts, itr=itr)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks-file', required=True, help='JSON file with task list')
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = 'cuda'

    with open(args.tasks_file) as f:
        task_list = json.load(f)

    print(f"GPU {args.gpu}: {len(task_list)} tasks across "
          f"{len(set(t['ckpt_path'] for t in task_list))} checkpoints", flush=True)

    current_model = None
    current_ckpt = None
    done = 0

    for task in task_list:
        ckpt_path = task['ckpt_path']

        if ckpt_path != current_ckpt:
            t0 = time.time()
            model, config = load_model(ckpt_path, device)
            current_model = model
            current_ckpt = ckpt_path
            itr = task.get('itr', 100000)
            print(f"  Loaded {os.path.basename(ckpt_path)} ({time.time()-t0:.1f}s)", flush=True)

        t0 = time.time()
        try:
            process_task(task, current_model, config, device, None, itr)
            dt = time.time() - t0
            done += 1
            # Print status as JSON for launcher to parse
            print(json.dumps({
                'status': 'done', 'task': task['name'],
                'gpu': args.gpu, 'elapsed': round(dt, 1),
                'progress': f'{done}/{len(task_list)}'
            }), flush=True)
        except Exception as e:
            done += 1
            print(json.dumps({
                'status': 'fail', 'task': task['name'],
                'gpu': args.gpu, 'error': str(e)
            }), flush=True)


if __name__ == '__main__':
    main()
