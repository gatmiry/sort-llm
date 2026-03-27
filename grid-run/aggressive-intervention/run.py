"""
Aggressive intervention: for the LN1 model (V128_B16_LR1e-02_MI60000),
intervene Layer 0 attention for ALL sorting-phase tokens simultaneously,
boosting attention to the first unsorted token by varying intensity.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import types
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model_analysis import GPT, GPTConfig

CKPT = os.path.join(os.path.dirname(__file__), '..',
    'outputs/V128_B16_LR1e-02_MI60000_LN1_E64_H1_L2',
    'V128_B16_LR1e-02_MI60000_LN1_E64_H1_L2_itr60000.pt')
DEVICE = 'cuda'
NUM_TRIALS = 500
INTENSITY_VALUES = [-0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5,
                    1.75, 2.0, 2.5, 3.0, 3.5, 4.0]
ATTN_LAYER = 0
OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt['config']
    config = GPTConfig(
        block_size=cfg['block_size'], vocab_size=cfg['vocab_size'],
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


def run_experiment():
    model, config, itr = load_model(CKPT, DEVICE)
    bs = config.block_size
    vs = config.vocab_size
    first_token_idx = 0  # position of the first token in unsorted part

    block = model.transformer.h[ATTN_LAYER]
    original_forward = block.c_attn.forward

    results = {}
    for intensity in INTENSITY_VALUES:
        successes = 0

        def make_new_forward(intens):
            def new_forward(self_attn, x, layer_n=-1):
                B, T, C = x.size()
                qkv = self_attn.c_attn(x)
                q, k, v = qkv.split(self_attn.n_embd, dim=2)
                q = q.view(B, T, self_attn.n_heads, C // self_attn.n_heads).transpose(1, 2)
                k = k.view(B, T, self_attn.n_heads, C // self_attn.n_heads).transpose(1, 2)
                v = v.view(B, T, self_attn.n_heads, C // self_attn.n_heads).transpose(1, 2)
                attn = q @ k.transpose(-1, -2) * 0.1 / (k.size(-1)) ** 0.5
                for pos in range(bs, 2 * bs):
                    attn[:, :, pos, first_token_idx] = attn[:, :, pos, first_token_idx] + intens
                attn = attn.masked_fill(self_attn.bias[:, :, :T, :T] == 0, float('-inf'))
                attn = F.softmax(attn, dim=-1)
                y = attn @ v
                y = y.transpose(1, 2).contiguous().view(B, T, C)
                y = self_attn.c_proj(y)
                return y
            return new_forward

        block.c_attn.forward = types.MethodType(make_new_forward(intensity), block.c_attn)

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
        print(f"  intensity={intensity:+.2f}: {rate:.4f} ({successes}/{NUM_TRIALS})", flush=True)

    return results, itr


def plot_results(results, itr):
    intensities = sorted(results.keys())
    rates = [results[i] for i in intensities]

    plt.figure(figsize=(5, 3.5))
    plt.plot(intensities, rates, marker='o', linewidth=1.8, markersize=5,
             color='#e6850e', label='With LayerNorm (LN1)')
    plt.xlabel('Intervention Intensity', fontsize=10)
    plt.ylabel('Full-sequence Success Rate', fontsize=10)
    plt.title(
        f'Aggressive Intervention — Layer 0, all sorting tokens\n'
        f'Boost attention to first unsorted token for every output position\n'
        f'V128 B16 LR1e-02 MI60000 ckpt={itr} ({NUM_TRIALS} trials)',
        fontsize=10, fontweight='bold')
    plt.legend(fontsize=8, loc='lower left')
    plt.grid(True, alpha=0.3)
    plt.xticks(intensities[::2], fontsize=8)
    plt.yticks(fontsize=8)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'aggressive_intensity_layer0.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {path}")


def main():
    print("Running aggressive intervention experiment...")
    results, itr = run_experiment()
    np.savez(os.path.join(OUT_DIR, 'results.npz'),
             intensities=np.array(sorted(results.keys())),
             success_rates=np.array([results[i] for i in sorted(results.keys())]),
             itr=itr)
    plot_results(results, itr)
    print("Done!")


if __name__ == '__main__':
    main()
