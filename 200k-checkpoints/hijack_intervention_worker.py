"""
GPU worker for hijack intervention experiment.
For each sorted position, boost a random unsorted number's attention with high
intensity, then record: current number, intervened-toward number, what the model
predicted, and the correct next number.
"""
import argparse
import os
import sys
import time
import types
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'grid-run'))
from model_analysis import GPT, GPTConfig

INTENSITY = 10.0


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


def run_trials(model, config, device, n_trials):
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

            candidates = []
            for i in range(bs):
                if unsorted[i].item() != correct_next:
                    candidates.append(i)
            if not candidates:
                continue

            boost_idx = candidates[torch.randint(len(candidates), (1,)).item()]
            boosted_number = unsorted[boost_idx].item()

            old_forward = attn_module.forward
            loc_capture = location
            bidx_capture = boost_idx
            mav_capture = main_attn_val

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

            attn_module.forward = types.MethodType(
                make_new_forward(loc_capture, bidx_capture, mav_capture), attn_module)

            with torch.no_grad():
                logits, _ = model(idx)
            predicted = torch.argmax(logits, dim=-1)[0, location].item()

            attn_module.forward = old_forward

            records.append((current_num, boosted_number, predicted, correct_next))

        if (trial + 1) % 500 == 0:
            print(f"  Trial {trial+1}/{n_trials}: {len(records)} records", flush=True)

    return np.array(records, dtype=np.int32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--trials', type=int, default=2000)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = 'cuda'

    print(f"GPU {args.gpu}: loading model...", flush=True)
    model, config = load_model(args.ckpt, device)

    print(f"GPU {args.gpu}: running {args.trials} trials...", flush=True)
    t0 = time.time()
    data = run_trials(model, config, device, args.trials)
    elapsed = time.time() - t0

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, data=data)
    print(f"GPU {args.gpu}: done in {elapsed:.0f}s, {len(data)} records", flush=True)


if __name__ == '__main__':
    main()
