"""Quick run: intensity up to 30 for the 1M checkpoint, layer 0, ub=60."""
import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'grid-run'))
from model_analysis import GPT, GPTConfig, GPTIntervention

CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
    'sortgpt_k16_methfixed_mlp1_L2_N256_E64_pos0_fln1_wd0p0_lr0p03_dseed1337_iseed1337__final.pt')
PLOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs',
    'plots_V256_B16_LR3e-2_MI1000000_E64_H1_L2_ds1337_is1337_ckpt1000000')

INTENSITIES = [1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 14.0, 18.0, 22.0, 26.0, 30.0]
UB = 60
MIN_VALID = 200
GPU = 0


def remap_state_dict(sd):
    new = {}
    for k, v in sd.items():
        nk = k
        for i in range(10):
            nk = nk.replace(f'transformer.h.{i}.attn.', f'transformer.h.{i}.c_attn.')
            nk = nk.replace(f'transformer.h.{i}.mlp.', f'transformer.h.{i}.c_fc.')
        new[nk] = v
    return new


def load_model(device):
    ckpt = torch.load(CKPT, map_location='cpu')
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


def get_batch(vs, bs, device):
    x = torch.randperm(vs)[:bs]
    vals, _ = torch.sort(x)
    return torch.cat((x, torch.tensor([vs]), vals), dim=0).unsqueeze(0).to(device)


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU)
    device = 'cuda'
    print("Loading model...", flush=True)
    model, config = load_model(device)
    bs = config.block_size
    vs = config.vocab_size
    location = bs + 5

    for layer in [0, 1]:
        print(f"\nLayer {layer}:", flush=True)
        rates, counts = [], []
        for intens in INTENSITIES:
            attempts, rounds = [], 0
            while len(attempts) < MIN_VALID and rounds < 3000:
                rounds += 1
                idx = get_batch(vs, bs, device)
                try:
                    im = GPTIntervention(model, idx)
                    im.intervent_attention(
                        attention_layer_num=layer, location=location,
                        unsorted_lb=UB, unsorted_ub=UB,
                        unsorted_lb_num=0, unsorted_ub_num=1,
                        unsorted_intensity_inc=intens,
                        sorted_lb=0, sorted_num=0, sorted_intensity_inc=0.0)
                    g, n = im.check_if_still_works()
                    attempts.append(g == n)
                    im.revert_attention(layer)
                except:
                    continue
            rate = sum(attempts) / len(attempts) if attempts else 0.0
            rates.append(rate)
            counts.append(len(attempts))
            print(f"  intensity={intens:5.1f}: success={rate:.4f} (n={len(attempts)})", flush=True)

        intensities = np.array(INTENSITIES)
        rates = np.array(rates)

        plt.figure(figsize=(5.5, 3.8))
        plt.plot(intensities, rates, marker='o', linewidth=1.5, markersize=5, color='#e6850e')
        plt.xlabel('Intervention Intensity', fontsize=10)
        plt.ylabel('Success Probability', fontsize=10)
        plt.title(f'Robustness to Attention Intervention (Layer {layer})  [ub={UB}]\n'
                  f'V=256  B=16  lr=0.03  iters=1000000  dseed=1337  iseed=1337',
                  fontsize=10, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.xticks(intensities[::2], fontsize=9)
        plt.yticks(fontsize=9)
        plt.ylim(0, 1.05)
        plt.tight_layout()
        out = os.path.join(PLOT_DIR, f'intensity_layer{layer}_ub60_high.png')
        plt.savefig(out, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {out}", flush=True)

    print("\nDone!", flush=True)


if __name__ == '__main__':
    main()
