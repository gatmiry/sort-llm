"""
Aggressive intervention on Layer 1 for both checkpoint configs.
Same procedure as Layer 0: boost attention to first unsorted token for
ALL sorting-phase tokens simultaneously.
"""
import sys, os, types, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model_analysis import GPT, GPTConfig

DEVICE = 'cuda'
NUM_TRIALS = 500
INTENSITY_VALUES = [-0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5,
                    1.75, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0]
CKPT_BASE = os.path.join(os.path.dirname(__file__), '..', 'outputs')
ATTN_LAYER = 1

CONFIGS = [
    'V512_B32_LR1e-02_MI20000',
]


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


def find_ckpt(config_base, ln):
    mi = int(re.match(r'V\d+_B\d+_LR[\d.e+-]+_MI(\d+)', config_base).group(1))
    dir_name = f"{config_base}_LN{ln}_E64_H1_L2"
    dir_path = os.path.join(CKPT_BASE, dir_name)
    for f in os.listdir(dir_path):
        if f.endswith(f'_itr{mi}.pt'):
            return os.path.join(dir_path, f)
    raise FileNotFoundError(f"No checkpoint in {dir_path}")


def get_batch(vocab_size, block_size):
    x = torch.randperm(vocab_size)[:block_size]
    vals, _ = torch.sort(x)
    return torch.cat((x, torch.tensor([vocab_size]), vals), dim=0).unsqueeze(0)


def run_aggressive(model, config, attn_layer):
    bs = config.block_size
    vs = config.vocab_size
    block = model.transformer.h[attn_layer]
    original_forward = block.c_attn.forward

    results = {}
    for intensity in INTENSITY_VALUES:
        successes = 0

        def make_fwd(intens):
            def new_forward(self_attn, x, layer_n=-1):
                B, T, C = x.size()
                qkv = self_attn.c_attn(x)
                q, k, v = qkv.split(self_attn.n_embd, dim=2)
                q = q.view(B, T, self_attn.n_heads, C // self_attn.n_heads).transpose(1, 2)
                k = k.view(B, T, self_attn.n_heads, C // self_attn.n_heads).transpose(1, 2)
                v = v.view(B, T, self_attn.n_heads, C // self_attn.n_heads).transpose(1, 2)
                attn = q @ k.transpose(-1, -2) * 0.1 / (k.size(-1)) ** 0.5
                for pos in range(bs, 2 * bs):
                    attn[:, :, pos, 0] = attn[:, :, pos, 0] + intens
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
            with torch.no_grad():
                logits, _ = model(idx)
            preds = torch.argmax(logits[0, bs:2*bs, :], dim=1)
            targets = idx[0, bs+1:]
            if (preds == targets).all():
                successes += 1
        block.c_attn.forward = original_forward
        rate = successes / NUM_TRIALS
        results[intensity] = rate
        print(f"    intensity={intensity:+.2f}: {rate:.4f} ({successes}/{NUM_TRIALS})", flush=True)
    return results


def main():
    for cfg_base in CONFIGS:
        m = re.match(r'V(\d+)_B(\d+)_LR([\d.e+-]+)_MI(\d+)', cfg_base)
        vs, bs, lr, mi = m.group(1), m.group(2), m.group(3), m.group(4)

        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               f"{cfg_base}_E64_H1_L2")
        os.makedirs(out_dir, exist_ok=True)

        ckpt_0 = find_ckpt(cfg_base, 0)
        ckpt_1 = find_ckpt(cfg_base, 1)
        model_0, cfg_0, itr_0 = load_model(ckpt_0, DEVICE)
        model_1, cfg_1, itr_1 = load_model(ckpt_1, DEVICE)
        itr = itr_0 or itr_1
        tag = f"vocab={vs}  block={bs}  lr={lr}  iters={mi}  ckpt={itr}"

        print(f"\n{'='*60}")
        print(f"  {cfg_base} — Aggressive Intervention Layer {ATTN_LAYER}")
        print(f"{'='*60}")

        all_data = {}
        for ln, model, config in [(0, model_0, cfg_0), (1, model_1, cfg_1)]:
            print(f"  LN{ln}:")
            results = run_aggressive(model, config, ATTN_LAYER)
            intensities = sorted(results.keys())
            rates = [results[i] for i in intensities]
            np.savez(os.path.join(out_dir, f'aggressive_layer1_LN{ln}.npz'),
                     intensities=np.array(intensities), success_rates=np.array(rates), itr=itr)
            all_data[ln] = (np.array(intensities), np.array(rates))

        plt.figure(figsize=(5, 3.5))
        plt.plot(all_data[0][0], all_data[0][1], marker='o', linewidth=1.5, markersize=5,
                 label='without layer norm', color='#1f77b4')
        plt.plot(all_data[1][0], all_data[1][1], marker='s', linewidth=1.5, markersize=5,
                 label='with layer norm', color='#ff7f0e')
        plt.xlabel('Intervention Intensity', fontsize=9)
        plt.ylabel('Full-sequence Success Rate', fontsize=9)
        plt.title(f'Aggressive Intervention — Layer {ATTN_LAYER}, all sorting tokens\n'
                  f'Boost attn to first unsorted token for every output position\n'
                  f'{tag} ({NUM_TRIALS} trials)', fontsize=9, fontweight='bold')
        plt.legend(fontsize=7, loc='lower left')
        plt.grid(True, alpha=0.3)
        plt.xticks(all_data[0][0][::2], fontsize=8)
        plt.yticks(fontsize=8)
        plt.ylim(0, 1.05)
        plt.tight_layout()
        path = os.path.join(out_dir, f'aggressive_intensity_layer{ATTN_LAYER}.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved {path}")

        del model_0, model_1
        torch.cuda.empty_cache()

    print("\nAll done!")


if __name__ == '__main__':
    main()
