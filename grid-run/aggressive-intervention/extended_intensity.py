"""
Standard per-token intervention (same as normal intensity plots) but with
extended intensity range up to 4.0. Produces compare_intensity plots for
V128_B16_LR1e-02_MI60000 with ub=60, layers 0 and 1.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model_analysis import GPT, GPTConfig, GPTIntervention

CKPT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
CKPT_LN0 = os.path.join(CKPT_DIR, 'V128_B16_LR1e-02_MI60000_LN0_E64_H1_L2',
                         'V128_B16_LR1e-02_MI60000_LN0_E64_H1_L2_itr60000.pt')
CKPT_LN1 = os.path.join(CKPT_DIR, 'V128_B16_LR1e-02_MI60000_LN1_E64_H1_L2',
                         'V128_B16_LR1e-02_MI60000_LN1_E64_H1_L2_itr60000.pt')
DEVICE = 'cuda'
MIN_VALID = 200
UB = 60
INTENSITY_VALUES = [-0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5,
                    1.75, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0]
OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt['config']
    config = GPTConfig(block_size=cfg['block_size'], vocab_size=cfg['vocab_size'],
                       with_layer_norm=cfg['with_layer_norm'])
    model = GPT(config)
    model.load_state_dict(ckpt['model'])
    model.to(device)
    model.eval()
    return model, config


def get_batch(vocab_size, block_size):
    x = torch.randperm(vocab_size)[:block_size]
    vals, _ = torch.sort(x)
    return torch.cat((x, torch.tensor([vocab_size]), vals), dim=0).unsqueeze(0)


def compute_intensity(model, config, device, attn_layer):
    bs = config.block_size
    vs = config.vocab_size
    location = bs + 5
    rates = []
    for intensity in INTENSITY_VALUES:
        attempts = []
        rounds = 0
        while len(attempts) < MIN_VALID and rounds < 2000:
            rounds += 1
            idx = get_batch(vs, bs).to(device)
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


def plot(intensities_0, rates_0, intensities_1, rates_1, layer, tag):
    plt.figure(figsize=(5, 3.5))
    plt.plot(intensities_0, rates_0, marker='o', linewidth=1.5, markersize=5,
             label='without layer norm', color='#1f77b4')
    plt.plot(intensities_1, rates_1, marker='s', linewidth=1.5, markersize=5,
             label='with layer norm', color='#ff7f0e')
    plt.xlabel('Intervention Intensity', fontsize=9)
    plt.ylabel('Success Probability', fontsize=9)
    plt.title(f'Robustness to Attention Intervention (Layer {layer})  [ub={UB}]\n'
              f'{tag}  (extended to 8.0)', fontsize=10)
    plt.legend(fontsize=7, loc='lower left')
    plt.grid(True, alpha=0.3)
    plt.xticks(intensities_0[::2], fontsize=8)
    plt.yticks(fontsize=8)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, f'compare_intensity_layer{layer}_ub{UB}_ext.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {path}")


def main():
    tag = 'vocab=128  block=16  lr=1e-02  iters=60000  ckpt=60000'
    model_0, cfg_0 = load_model(CKPT_LN0, DEVICE)
    model_1, cfg_1 = load_model(CKPT_LN1, DEVICE)

    for layer in [0, 1]:
        print(f"\n=== Layer {layer} ===")
        print(f"  LN0:")
        i0, r0 = compute_intensity(model_0, cfg_0, DEVICE, layer)
        print(f"  LN1:")
        i1, r1 = compute_intensity(model_1, cfg_1, DEVICE, layer)
        plot(i0, r0, i1, r1, layer, tag)

    print("\nDone!")


if __name__ == '__main__':
    main()
