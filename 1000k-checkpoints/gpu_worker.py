"""
GPU worker for 1000k-checkpoint analysis.
Processes all task types on a single GPU: baseline, ablation, cinclogits,
intensity (various ub), asymmetric intensity, hijack, separator/random.
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'grid-run'))
from model_analysis import GPT, GPTConfig, GPTIntervention


def remap_state_dict(sd):
    new_sd = {}
    for key, val in sd.items():
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


def compute_ablation(model, config, device, skip_layer, num_trials=500):
    bs = config.block_size
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
            idx = get_batch(config.vocab_size, bs, device)
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


def compute_intensity(model, config, device, attn_layer, ub=5, lb=None,
                      ub_num=1, lb_num=0, min_valid=200):
    if lb is None:
        lb = ub
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
                    unsorted_lb=lb, unsorted_ub=ub,
                    unsorted_lb_num=lb_num, unsorted_ub_num=ub_num,
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


def compute_hijack(model, config, device, n_trials=2000):
    """Hijack intervention on layer 0. Returns array of (current, boosted, predicted, correct)."""
    INTENSITY = 10.0
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


def compute_separator_random(model, config, device, n_trials=1000):
    """Separator-attention and random-target intervention on layer 0."""
    INTENSITIES = [2.0, 6.0, 10.0]
    UB_STANDARD = 60
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

            attn_row = attn_layer0[sorted_loc, :sorted_loc + 1]
            max_attn_pos = attn_row.argmax().item()
            attends_to_sep = (max_attn_pos == sep_pos)

            for intensity in INTENSITIES:
                if attends_to_sep:
                    try:
                        im = GPTIntervention(model, idx)
                        im.intervent_attention(
                            attention_layer_num=0, location=sorted_loc,
                            unsorted_lb=UB_STANDARD, unsorted_ub=UB_STANDARD,
                            unsorted_lb_num=0, unsorted_ub_num=1,
                            unsorted_intensity_inc=intensity,
                            sorted_lb=0, sorted_num=0, sorted_intensity_inc=0.0)
                        g, n = im.check_if_still_works()
                        im.revert_attention(0)
                        sep_records.append((number_val, intensity, int(g == n)))
                    except:
                        pass

                try:
                    im = GPTIntervention(model, idx)
                    im.intervent_attention(
                        attention_layer_num=0, location=sorted_loc,
                        unsorted_lb=0, unsorted_ub=vs,
                        unsorted_lb_num=0, unsorted_ub_num=1,
                        unsorted_intensity_inc=intensity,
                        sorted_lb=0, sorted_num=0, sorted_intensity_inc=0.0)
                    g, n = im.check_if_still_works()
                    im.revert_attention(0)
                    rand_records.append((number_val, intensity, int(g == n)))
                except:
                    try:
                        im = GPTIntervention(model, idx)
                        im.intervent_attention(
                            attention_layer_num=0, location=sorted_loc,
                            unsorted_lb=vs, unsorted_ub=0,
                            unsorted_lb_num=1, unsorted_ub_num=0,
                            unsorted_intensity_inc=intensity,
                            sorted_lb=0, sorted_num=0, sorted_intensity_inc=0.0)
                        g, n = im.check_if_still_works()
                        im.revert_attention(0)
                        rand_records.append((number_val, intensity, int(g == n)))
                    except:
                        pass

    sep = np.array(sep_records, dtype=np.int32) if sep_records else np.empty((0, 3), dtype=np.int32)
    rand = np.array(rand_records, dtype=np.int32) if rand_records else np.empty((0, 3), dtype=np.int32)
    return sep, rand


def process_task(task, model, config, device):
    task_type = task['type']
    out_path = task['out']
    if os.path.exists(out_path):
        return True

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    itr = task.get('itr', 0)

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

    elif task_type == 'intensity_asym':
        intensities, rates, counts = compute_intensity(
            model, config, device, task['layer'],
            ub=task['unsorted_ub'], lb=task['unsorted_lb'],
            ub_num=task['unsorted_ub_num'], lb_num=task['unsorted_lb_num'])
        np.savez(out_path, intensities=intensities, success_rates=rates,
                 counts=counts, itr=itr)

    elif task_type == 'hijack':
        data = compute_hijack(model, config, device, n_trials=task.get('trials', 2000))
        np.savez(out_path, data=data)

    elif task_type == 'separator_random':
        sep, rand = compute_separator_random(model, config, device,
                                             n_trials=task.get('trials', 1000))
        np.savez(out_path, sep_data=sep, rand_data=rand)

    return True


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
            print(f"  Loaded {os.path.basename(ckpt_path)} ({time.time()-t0:.1f}s)", flush=True)

        t0 = time.time()
        try:
            process_task(task, current_model, config, device)
            dt = time.time() - t0
            done += 1
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

    print(f"GPU {args.gpu}: all done ({done}/{len(task_list)})", flush=True)


if __name__ == '__main__':
    main()
