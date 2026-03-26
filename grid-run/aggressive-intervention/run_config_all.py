"""
Combined aggressive-intervention script (BATCHED for speed).
Runs all three experiment types for a given config.

Experiments:
  1. Aggressive intervention (boost attn to first-unsorted pos) — layers 0 & 1
  2. Aggressive SEP intervention (boost attn to SEP pos) — layers 0 & 1
  3. Extended standard per-token intervention (ub=60) — layers 0 & 1

Usage:
  python run_config_all.py V512_B32_LR1e-02_MI40000 --device cuda
"""
import sys, os, argparse, types, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model_analysis import GPT, GPTConfig, GPTIntervention, CasualSelfAttention


def _patched_attn_forward(self, x, layer_n=-1):
    B, T, C = x.size()
    qkv = self.c_attn(x)
    q, k, v = qkv.split(self.n_embd, dim=2)
    q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
    k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
    v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
    attn = q @ k.transpose(-1, -2) * 0.1 / (k.size(-1)) ** 0.5
    attn = attn.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
    self.raw_attn = attn.clone().detach()
    attn = F.softmax(attn, dim=-1)
    self.attn = attn.clone().detach()
    y = attn @ v
    y = y.transpose(1, 2).contiguous().view(B, T, C)
    y = self.c_proj(y)
    return y


CasualSelfAttention.forward = _patched_attn_forward

NUM_TRIALS = 500
BATCH_SIZE = 128
MIN_VALID = 200
UB = 60
INTENSITY_VALUES = [-4.0, -3.5, -3.0, -2.5, -2.0, -1.75, -1.5, -1.25, -1.0,
                    -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5,
                    1.75, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0]
CKPT_BASE = os.path.join(os.path.dirname(__file__), '..', 'outputs')


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt['config']
    config = GPTConfig(block_size=cfg['block_size'], vocab_size=cfg['vocab_size'],
                       with_layer_norm=cfg['with_layer_norm'])
    model = GPT(config)
    model.load_state_dict(ckpt['model'])
    model.to(device)
    model.eval()
    return model, config, ckpt.get('itr', None)


def get_batch(vocab_size, block_size, batch_size, device):
    """Generate a batch of B samples directly on device."""
    samples = []
    for _ in range(batch_size):
        x = torch.randperm(vocab_size)[:block_size]
        vals, _ = torch.sort(x)
        samples.append(torch.cat((x, torch.tensor([vocab_size]), vals), dim=0))
    return torch.stack(samples).to(device)


def find_ckpt(config_base, ln):
    mi = int(re.match(r'V\d+_B\d+_LR[\d.e+-]+_MI(\d+)', config_base).group(1))
    dir_name = f"{config_base}_LN{ln}_E64_H1_L2"
    dir_path = os.path.join(CKPT_BASE, dir_name)
    for f in os.listdir(dir_path):
        if f.endswith(f'_itr{mi}.pt'):
            return os.path.join(dir_path, f)
    raise FileNotFoundError(f"No checkpoint in {dir_path}")


# ── Aggressive intervention (boost attn to first unsorted token) ─────────────

def run_aggressive(model, config, attn_layer, device):
    bs = config.block_size
    vs = config.vocab_size

    block = model.transformer.h[attn_layer]
    original_forward = block.c_attn.forward
    current_batch = [None]

    results = {}
    for intensity in INTENSITY_VALUES:
        correct_tokens = 0
        total_tokens = 0

        def make_fwd(intens):
            def new_forward(self_attn, x, layer_n=-1):
                B, T, C = x.size()
                qkv = self_attn.c_attn(x)
                q, k, v = qkv.split(self_attn.n_embd, dim=2)
                q = q.view(B, T, self_attn.n_heads, C // self_attn.n_heads).transpose(1, 2)
                k = k.view(B, T, self_attn.n_heads, C // self_attn.n_heads).transpose(1, 2)
                v = v.view(B, T, self_attn.n_heads, C // self_attn.n_heads).transpose(1, 2)
                att = q @ k.transpose(-1, -2) * 0.1 / (k.size(-1)) ** 0.5

                idx_t = current_batch[0]
                first_vals = idx_t[:, 0]
                targets_all = idx_t[:, bs + 1: 2 * bs + 1]
                unsorted_all = idx_t[:, :bs]

                lookup = torch.zeros(B, vs + 1, dtype=torch.long, device=x.device)
                batch_idx = torch.arange(B, device=x.device).unsqueeze(1).expand_as(unsorted_all)
                pos_idx = torch.arange(bs, device=x.device).unsqueeze(0).expand_as(unsorted_all)
                lookup[batch_idx, unsorted_all] = pos_idx

                target_locs = lookup[torch.arange(B, device=x.device).unsqueeze(1).expand_as(targets_all),
                                     targets_all]

                out_positions = torch.arange(bs, 2 * bs, device=x.device)
                b_range = torch.arange(B, device=x.device)

                main_attn_vals = att[
                    b_range[:, None, None],
                    torch.arange(self_attn.n_heads, device=x.device)[None, :, None],
                    out_positions[None, None, :],
                    target_locs[:, None, :]
                ]

                mask = (targets_all != first_vals[:, None])
                mask_exp = mask[:, None, :].expand_as(main_attn_vals)

                att_slice = att[:, :, bs:2*bs, 0].clone()
                att_slice[mask_exp] = main_attn_vals[mask_exp] + intens
                att[:, :, bs:2*bs, 0] = att_slice

                att = att.masked_fill(self_attn.bias[:, :, :T, :T] == 0, float('-inf'))
                att = F.softmax(att, dim=-1)
                y = att @ v
                y = y.transpose(1, 2).contiguous().view(B, T, C)
                y = self_attn.c_proj(y)
                return y
            return new_forward

        block.c_attn.forward = types.MethodType(make_fwd(intensity), block.c_attn)

        remaining = NUM_TRIALS
        while remaining > 0:
            chunk = min(remaining, BATCH_SIZE)
            idx = get_batch(vs, bs, chunk, device)
            current_batch[0] = idx
            with torch.no_grad():
                logits, _ = model(idx)
            preds = torch.argmax(logits[:, bs:2*bs, :], dim=2)
            targets = idx[:, bs+1:]
            correct_tokens += (preds == targets).sum().item()
            total_tokens += targets.numel()
            remaining -= chunk

        block.c_attn.forward = original_forward
        rate = correct_tokens / total_tokens
        results[intensity] = rate
        print(f"    intensity={intensity:+.2f}: {rate:.4f} ({correct_tokens}/{total_tokens})", flush=True)
    return results


# ── Aggressive SEP intervention (boost attn to SEP token) ────────────────────

def run_aggressive_sep(model, config, attn_layer, device):
    bs = config.block_size
    vs = config.vocab_size
    sep_pos = bs

    block = model.transformer.h[attn_layer]
    original_forward = block.c_attn.forward
    current_batch = [None]

    results = {}
    for intensity in INTENSITY_VALUES:
        correct_tokens = 0
        total_tokens = 0

        def make_fwd(intens):
            def new_forward(self_attn, x, layer_n=-1):
                B, T, C = x.size()
                qkv = self_attn.c_attn(x)
                q, k, v = qkv.split(self_attn.n_embd, dim=2)
                q = q.view(B, T, self_attn.n_heads, C // self_attn.n_heads).transpose(1, 2)
                k = k.view(B, T, self_attn.n_heads, C // self_attn.n_heads).transpose(1, 2)
                v = v.view(B, T, self_attn.n_heads, C // self_attn.n_heads).transpose(1, 2)
                att = q @ k.transpose(-1, -2) * 0.1 / (k.size(-1)) ** 0.5

                idx_t = current_batch[0]
                targets_all = idx_t[:, bs + 1: 2 * bs + 1]
                unsorted_all = idx_t[:, :bs]

                lookup = torch.zeros(B, vs + 1, dtype=torch.long, device=x.device)
                batch_idx = torch.arange(B, device=x.device).unsqueeze(1).expand_as(unsorted_all)
                pos_idx = torch.arange(bs, device=x.device).unsqueeze(0).expand_as(unsorted_all)
                lookup[batch_idx, unsorted_all] = pos_idx

                target_locs = lookup[torch.arange(B, device=x.device).unsqueeze(1).expand_as(targets_all),
                                     targets_all]

                out_positions = torch.arange(bs, 2 * bs, device=x.device)
                b_range = torch.arange(B, device=x.device)

                main_attn_vals = att[
                    b_range[:, None, None],
                    torch.arange(self_attn.n_heads, device=x.device)[None, :, None],
                    out_positions[None, None, :],
                    target_locs[:, None, :]
                ]

                att[:, :, bs:2*bs, sep_pos] = main_attn_vals + intens

                att = att.masked_fill(self_attn.bias[:, :, :T, :T] == 0, float('-inf'))
                att = F.softmax(att, dim=-1)
                y = att @ v
                y = y.transpose(1, 2).contiguous().view(B, T, C)
                y = self_attn.c_proj(y)
                return y
            return new_forward

        block.c_attn.forward = types.MethodType(make_fwd(intensity), block.c_attn)

        remaining = NUM_TRIALS
        while remaining > 0:
            chunk = min(remaining, BATCH_SIZE)
            idx = get_batch(vs, bs, chunk, device)
            current_batch[0] = idx
            with torch.no_grad():
                logits, _ = model(idx)
            preds = torch.argmax(logits[:, bs:2*bs, :], dim=2)
            targets = idx[:, bs+1:]
            correct_tokens += (preds == targets).sum().item()
            total_tokens += targets.numel()
            remaining -= chunk

        block.c_attn.forward = original_forward
        rate = correct_tokens / total_tokens
        results[intensity] = rate
        print(f"    intensity={intensity:+.2f}: {rate:.4f} ({correct_tokens}/{total_tokens})", flush=True)
    return results


# ── Extended standard per-token intervention (NOT batched — uses GPTIntervention) ─

def run_standard_intensity(model, config, attn_layer, device):
    bs = config.block_size
    vs = config.vocab_size
    ub_eff = min(UB, bs)
    location = bs + 5
    rates = []
    for intensity in INTENSITY_VALUES:
        attempts = []
        rounds = 0
        while len(attempts) < MIN_VALID and rounds < 2000:
            rounds += 1
            x = torch.randperm(vs)[:bs]
            vals, _ = torch.sort(x)
            idx = torch.cat((x, torch.tensor([vs]), vals), dim=0).unsqueeze(0).to(device)
            try:
                im = GPTIntervention(model, idx)
                im.intervent_attention(
                    attention_layer_num=attn_layer, location=location,
                    unsorted_lb=ub_eff, unsorted_ub=ub_eff,
                    unsorted_lb_num=0, unsorted_ub_num=1,
                    unsorted_intensity_inc=intensity,
                    sorted_lb=0, sorted_num=0, sorted_intensity_inc=0.0)
                new_gen, next_num = im.check_if_still_works()
                attempts.append(new_gen == next_num)
                im.revert_attention(attn_layer)
            except:
                continue
        rate = sum(attempts) / len(attempts) if attempts else 0.0
        rates.append(rate)
        print(f"    intensity={intensity:+.2f}: {rate:.4f} ({len(attempts)} valid)", flush=True)
    return np.array(INTENSITY_VALUES), np.array(rates)


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_pair(data_ln0, data_ln1, out_path, title, ylabel='Token Success Rate'):
    plt.figure(figsize=(5, 3.5))
    plt.plot(data_ln0[0], data_ln0[1], marker='o', linewidth=1.5, markersize=5,
             label='without layer norm', color='#1f77b4')
    plt.plot(data_ln1[0], data_ln1[1], marker='s', linewidth=1.5, markersize=5,
             label='with layer norm', color='#ff7f0e')
    plt.xlabel('Intervention Intensity', fontsize=9)
    plt.ylabel(ylabel, fontsize=9)
    plt.title(title, fontsize=9, fontweight='bold')
    plt.legend(fontsize=7, loc='lower left')
    plt.grid(True, alpha=0.3)
    plt.xticks(data_ln0[0][::2], fontsize=8)
    plt.yticks(fontsize=8)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='e.g. V512_B32_LR1e-02_MI40000')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--aggressive-only', action='store_true',
                        help='Only run aggressive+SEP, skip extended standard')
    args = parser.parse_args()
    cfg_name = args.config
    device = args.device

    m = re.match(r'V(\d+)_B(\d+)_LR([\d.e+-]+)_MI(\d+)', cfg_name)
    vs, bs, lr, mi = m.group(1), m.group(2), m.group(3), m.group(4)

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           f"{cfg_name}_E64_H1_L2")
    os.makedirs(out_dir, exist_ok=True)

    ckpt_0 = find_ckpt(cfg_name, 0)
    ckpt_1 = find_ckpt(cfg_name, 1)
    model_0, cfg_0, itr_0 = load_model(ckpt_0, device)
    model_1, cfg_1, itr_1 = load_model(ckpt_1, device)
    itr = itr_0 if itr_0 == itr_1 else max(itr_0, itr_1)
    tag = f"vocab={vs}  block={bs}  lr={lr}  iters={mi}  ckpt={itr}"

    # ── 1. Aggressive intervention (layers 0 & 1) ────────────────────────
    for attn_layer in [0, 1]:
        png = os.path.join(out_dir, f'aggressive_intensity_layer{attn_layer}.png')
        if os.path.exists(png):
            print(f"SKIP aggressive layer {attn_layer} (already exists)")
            continue
        print(f"\n=== Aggressive Intervention (Layer {attn_layer}) ===")
        all_data = {}
        for ln_val, model, config in [(0, model_0, cfg_0), (1, model_1, cfg_1)]:
            print(f"  LN{ln_val}:")
            results = run_aggressive(model, config, attn_layer, device)
            intensities = sorted(results.keys())
            rates = [results[i] for i in intensities]
            np.savez(os.path.join(out_dir, f'aggressive_layer{attn_layer}_LN{ln_val}.npz'),
                     intensities=np.array(intensities), token_success_rates=np.array(rates), itr=itr)
            all_data[ln_val] = (np.array(intensities), np.array(rates))
        plot_pair(all_data[0], all_data[1], png,
                  f'Aggressive Intervention — Layer {attn_layer}, all sorting tokens\n'
                  f'Set attn to pos-0 = attn(correct) + intensity, all positions\n'
                  f'{tag} ({NUM_TRIALS} trials)')

    # ── 2. Aggressive SEP intervention (layers 0 & 1) ────────────────────
    for attn_layer in [0, 1]:
        png = os.path.join(out_dir, f'aggressive_sep_intensity_layer{attn_layer}.png')
        if os.path.exists(png):
            print(f"SKIP aggressive SEP layer {attn_layer} (already exists)")
            continue
        print(f"\n=== Aggressive SEP Intervention (Layer {attn_layer}) ===")
        all_data = {}
        for ln_val, model, config in [(0, model_0, cfg_0), (1, model_1, cfg_1)]:
            print(f"  LN{ln_val}:")
            results = run_aggressive_sep(model, config, attn_layer, device)
            intensities = sorted(results.keys())
            rates = [results[i] for i in intensities]
            np.savez(os.path.join(out_dir, f'aggressive_sep_layer{attn_layer}_LN{ln_val}.npz'),
                     intensities=np.array(intensities), success_rates=np.array(rates), itr=itr)
            all_data[ln_val] = (np.array(intensities), np.array(rates))
        plot_pair(all_data[0], all_data[1], png,
                  f'Aggressive SEP Intervention — Layer {attn_layer}, all sorting tokens\n'
                  f'Set attn to SEP = attn(correct) + intensity, all positions\n'
                  f'{tag} ({NUM_TRIALS} trials)')

    if args.aggressive_only:
        print(f"\nDone (aggressive+SEP only) for {cfg_name}!")
        return

    # ── 3. Extended standard intervention (layers 0 & 1) ─────────────────
    for layer in [0, 1]:
        ub_eff = min(UB, int(bs))
        png = os.path.join(out_dir, f'compare_intensity_layer{layer}_ub{ub_eff}_ext.png')
        if os.path.exists(png):
            print(f"SKIP extended standard layer {layer} (already exists)")
            continue
        print(f"\n=== Extended Standard Intervention (Layer {layer}, ub={ub_eff}) ===")
        print(f"  LN0:")
        i0, r0 = run_standard_intensity(model_0, cfg_0, layer, device)
        print(f"  LN1:")
        i1, r1 = run_standard_intensity(model_1, cfg_1, layer, device)
        plot_pair((i0, r0), (i1, r1), png,
                  f'Robustness to Attention Intervention (Layer {layer})  [ub={ub_eff}]\n'
                  f'{tag}  (extended to 8.0)',
                  ylabel='Success Probability')

    print(f"\nAll done for {cfg_name}!")


if __name__ == '__main__':
    main()
