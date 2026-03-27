"""
Generate compare_intensity_layer0_ub30_ext.png for V512_B32_LR1e-03_MI10000.
Same as compare_intensity_layer0_ub60_ext.png but with ub/lb=30 instead of 60.
"""
import sys, os, types, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model_analysis import GPT, GPTConfig, GPTIntervention

DEVICE = 'cpu'
NUM_TRIALS = 500
MIN_VALID = 200
UB = 30
INTENSITY_VALUES = [-4.0, -3.5, -3.0, -2.5, -2.0, -1.75, -1.5, -1.25, -1.0,
                    -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5,
                    1.75, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0]
CKPT_BASE = os.path.join(os.path.dirname(__file__), '..', 'outputs')
CONFIG_BASE = 'V512_B32_LR1e-03_MI10000'


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
    m = re.match(r'V(\d+)_B(\d+)_LR([\d.e+-]+)_MI(\d+)', CONFIG_BASE)
    vs, bs, lr, mi = m.group(1), m.group(2), m.group(3), m.group(4)

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           f"{CONFIG_BASE}_E64_H1_L2")
    os.makedirs(out_dir, exist_ok=True)

    ckpt_0 = find_ckpt(CONFIG_BASE, 0)
    ckpt_1 = find_ckpt(CONFIG_BASE, 1)
    model_0, cfg_0, itr_0 = load_model(ckpt_0, DEVICE)
    model_1, cfg_1, itr_1 = load_model(ckpt_1, DEVICE)
    itr = itr_0 if itr_0 == itr_1 else max(itr_0, itr_1)
    tag = f"vocab={vs}  block={bs}  lr={lr}  iters={mi}  ckpt={itr}"

    layer = 0
    print(f"\n=== Standard Intervention (Layer {layer}, ub={UB}) ===")
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

    print("\nDone!")


if __name__ == '__main__':
    main()
