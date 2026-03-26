"""
Runs both aggressive intervention (all sorting tokens, Layer 0) and
extended standard intervention (per-token, ub=60, layers 0&1) for a
given config. Outputs go into a subfolder named after the config.

Usage:
  python run_config.py V128_B32_LR1e-02_MI40000
"""
import sys, os, argparse, types, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model_analysis import GPT, GPTConfig, GPTIntervention

DEVICE = 'cpu'
NUM_TRIALS = 500
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


def get_batch(vocab_size, block_size):
    x = torch.randperm(vocab_size)[:block_size]
    vals, _ = torch.sort(x)
    return torch.cat((x, torch.tensor([vocab_size]), vals), dim=0).unsqueeze(0)


def find_ckpt(config_base, ln):
    m = re.match(r'V(\d+)_B(\d+)_LR([\d.e+-]+)_MI(\d+)', config_base)
    mi = int(m.group(4))
    dir_name = f"{config_base}_LN{ln}_E64_H1_L2"
    dir_path = os.path.join(CKPT_BASE, dir_name)
    target = f"_itr{mi}.pt"
    for f in os.listdir(dir_path):
        if f.endswith(target):
            return os.path.join(dir_path, f)
    raise FileNotFoundError(f"No checkpoint in {dir_path}")


# ── Aggressive intervention ──────────────────────────────────────────────────

def run_aggressive(model, config, attn_layer):
    bs = config.block_size
    vs = config.vocab_size

    block = model.transformer.h[attn_layer]
    original_forward = block.c_attn.forward
    current_idx = [None]

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
                attn = q @ k.transpose(-1, -2) * 0.1 / (k.size(-1)) ** 0.5

                idx_tensor = current_idx[0]
                first_token_val = idx_tensor[0, 0].item()

                for pos in range(bs, 2 * bs):
                    next_number = idx_tensor[0, pos + 1].item()
                    if first_token_val == next_number:
                        continue
                    next_num_loc = (idx_tensor[0, :bs] == next_number).nonzero(as_tuple=True)[0][0].item()
                    main_attn_val = attn[:, :, pos, next_num_loc].clone()
                    attn[:, :, pos, 0] = main_attn_val + intens

                attn = attn.masked_fill(self_attn.bias[:, :, :T, :T] == 0, float('-inf'))
                attn = F.softmax(attn, dim=-1)
                y = attn @ v
                y = y.transpose(1, 2).contiguous().view(B, T, C)
                y = self_attn.c_proj(y)
                return y
            return new_forward

        block.c_attn.forward = types.MethodType(make_fwd(intensity), block.c_attn)
        for _ in range(NUM_TRIALS):
            idx = get_batch(vs, bs).to(DEVICE)
            current_idx[0] = idx
            with torch.no_grad():
                logits, _ = model(idx)
            preds = torch.argmax(logits[0, bs:2*bs, :], dim=1)
            targets = idx[0, bs+1:]
            correct_tokens += (preds == targets).sum().item()
            total_tokens += len(targets)
        block.c_attn.forward = original_forward
        rate = correct_tokens / total_tokens
        results[intensity] = rate
        print(f"    intensity={intensity:+.2f}: {rate:.4f} ({correct_tokens}/{total_tokens})", flush=True)

    return results


def run_standard_intensity(model, config, attn_layer):
    bs = config.block_size
    vs = config.vocab_size
    location = bs + 5
    rates = []
    for intensity in INTENSITY_VALUES:
        attempts = []
        rounds = 0
        while len(attempts) < MIN_VALID and rounds < 2000:
            rounds += 1
            idx = get_batch(vs, bs).to(DEVICE)
            try:
                im = GPTIntervention(model, idx)
                im.intervent_attention(
                    attention_layer_num=attn_layer, location=location,
                    unsorted_lb=UB, unsorted_ub=UB,
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='e.g. V128_B32_LR1e-02_MI40000')
    parser.add_argument('--aggressive-only', action='store_true',
                        help='Only run aggressive intervention, skip compare_intensity')
    args = parser.parse_args()
    cfg = args.config

    m = re.match(r'V(\d+)_B(\d+)_LR([\d.e+-]+)_MI(\d+)', cfg)
    vs, bs, lr, mi = m.group(1), m.group(2), m.group(3), m.group(4)

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           f"{cfg}_E64_H1_L2")
    os.makedirs(out_dir, exist_ok=True)

    ckpt_0 = find_ckpt(cfg, 0)
    ckpt_1 = find_ckpt(cfg, 1)
    model_0, cfg_0, itr_0 = load_model(ckpt_0, DEVICE)
    model_1, cfg_1, itr_1 = load_model(ckpt_1, DEVICE)
    itr = itr_0 if itr_0 == itr_1 else max(itr_0, itr_1)
    tag = f"vocab={vs}  block={bs}  lr={lr}  iters={mi}  ckpt={itr}"

    # ── Aggressive intervention (Layers 0 & 1, both models) ───────────────
    for attn_layer in [0, 1]:
        print(f"\n=== Aggressive Intervention (Layer {attn_layer}) ===")
        all_data = {}
        for ln, model, config in [(0, model_0, cfg_0), (1, model_1, cfg_1)]:
            print(f"  LN{ln}:")
            results = run_aggressive(model, config, attn_layer)
            intensities = sorted(results.keys())
            rates = [results[i] for i in intensities]
            np.savez(os.path.join(out_dir, f'aggressive_layer{attn_layer}_LN{ln}.npz'),
                     intensities=np.array(intensities), token_success_rates=np.array(rates), itr=itr)
            all_data[ln] = (np.array(intensities), np.array(rates))

        plt.figure(figsize=(5, 3.5))
        plt.plot(all_data[0][0], all_data[0][1], marker='o', linewidth=1.5, markersize=5,
                 label='without layer norm', color='#1f77b4')
        plt.plot(all_data[1][0], all_data[1][1], marker='s', linewidth=1.5, markersize=5,
                 label='with layer norm', color='#ff7f0e')
        plt.xlabel('Intervention Intensity', fontsize=9)
        plt.ylabel('Token Success Rate', fontsize=9)
        plt.title(f'Aggressive Intervention — Layer {attn_layer}, all sorting tokens\n'
                  f'Set attn to pos-0 = attn(correct) + intensity, all positions\n'
                  f'{tag} ({NUM_TRIALS} trials)', fontsize=9, fontweight='bold')
        plt.legend(fontsize=7, loc='lower left')
        plt.grid(True, alpha=0.3)
        plt.xticks(all_data[0][0][::2], fontsize=8)
        plt.yticks(fontsize=8)
        plt.ylim(0, 1.05)
        plt.tight_layout()
        path = os.path.join(out_dir, f'aggressive_intensity_layer{attn_layer}.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved {path}")

    if args.aggressive_only:
        print("\nDone (aggressive only)!")
        return

    # ── Extended standard intervention (layers 0 & 1) ────────────────────
    for layer in [0, 1]:
        print(f"\n=== Extended Standard Intervention (Layer {layer}, ub={UB}) ===")
        print(f"  LN0:")
        i0, r0 = run_standard_intensity(model_0, cfg_0, layer)
        print(f"  LN1:")
        i1, r1 = run_standard_intensity(model_1, cfg_1, layer)

        plt.figure(figsize=(5, 3.5))
        plt.plot(i0, r0, marker='o', linewidth=1.5, markersize=5,
                 label='without layer norm', color='#1f77b4')
        plt.plot(i1, r1, marker='s', linewidth=1.5, markersize=5,
                 label='with layer norm', color='#ff7f0e')
        plt.xlabel('Intervention Intensity', fontsize=9)
        plt.ylabel('Success Probability', fontsize=9)
        plt.title(f'Robustness to Attention Intervention (Layer {layer})  [ub={UB}]\n'
                  f'{tag}  (extended to 8.0)', fontsize=10)
        plt.legend(fontsize=7, loc='lower left')
        plt.grid(True, alpha=0.3)
        plt.xticks(i0[::2], fontsize=8)
        plt.yticks(fontsize=8)
        plt.ylim(0, 1.05)
        plt.tight_layout()
        path = os.path.join(out_dir, f'compare_intensity_layer{layer}_ub{UB}_ext.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved {path}")

    print("\nAll done!")


if __name__ == '__main__':
    main()
