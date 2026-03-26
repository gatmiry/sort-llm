"""
Generate analysis plots for a grid-run config group (LN0 vs LN1).
Produces four plots per config:
  1. compare_cinclogits_layer0.png – Incorrect-score fraction & logit correction (layer 0)
  2. compare_cinclogits_layer1.png – Incorrect-score fraction & logit correction (layer 1)
  3. compare_intensity_layer0.png  – Intervention robustness (layer 0)
  4. compare_intensity_layer1.png  – Intervention robustness (layer 1)

Usage:
  python analyze_run.py --vocab_size 64 --block_size 16 --lr 1e-03 --max_iters 10000
  # Loads the final checkpoint for both LN0 and LN1, saves plots into
  # outputs/plots_vocab64_block16_lr1e-03_iters10000/
"""
import argparse
import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed

from model_analysis import GPT, GPTConfig, GPTIntervention

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_BASE = os.path.join(SCRIPT_DIR, 'outputs')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--vocab_size', type=int, required=True)
    p.add_argument('--block_size', type=int, required=True)
    p.add_argument('--lr', type=str, required=True)
    p.add_argument('--max_iters', type=int, required=True)
    p.add_argument('--device', type=str, default='cuda')
    return p.parse_args()


def find_final_ckpt(run_dir, max_iters):
    target = f"_itr{max_iters}.pt"
    for f in os.listdir(run_dir):
        if f.endswith(target):
            return os.path.join(run_dir, f)
    return None


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt['config']
    config = GPTConfig(
        block_size=cfg['block_size'],
        vocab_size=cfg['vocab_size'],
        with_layer_norm=cfg['with_layer_norm'],
    )
    model = GPT(config)
    model.load_state_dict(ckpt['model'])
    model.to(device)
    model.eval()
    itr = ckpt.get('itr', None)
    return model, config, itr


def get_batch(vocab_size, block_size):
    x = torch.randperm(vocab_size)[:block_size]
    vals, _ = torch.sort(x)
    return torch.cat((x, torch.tensor([vocab_size]), vals), dim=0).unsqueeze(0)


# ── Plot 1: compare_cinclogits ──────────────────────────────────────────────

def compute_cinclogits(model, config, device, num_tries=100, attn_layer=0):
    block_size = config.block_size
    vocab_size = config.vocab_size
    acc_clogit_icscore = np.zeros(block_size)
    acc_iclogit_icscore = np.zeros(block_size)

    for _ in range(num_tries):
        idx = get_batch(vocab_size, block_size).to(device)
        with torch.no_grad():
            logits, _ = model(idx)

        is_correct = (torch.argmax(logits[0, block_size:2 * block_size, :], dim=1)
                      == idx[0, block_size + 1:])
        attn_weights = model.transformer.h[attn_layer].c_attn.attn

        for j in range(block_size, 2 * block_size):
            max_score = float('-inf')
            max_score_num = -1
            for k in range(0, 2 * block_size + 1):
                score = attn_weights[j, k].item()
                if score > max_score:
                    max_score = score
                    max_score_num = idx[0, k].item()
            score_correct = (max_score_num == idx[0, j + 1].item())
            pos = j - block_size
            logit_correct = is_correct[pos].item()
            if logit_correct and not score_correct:
                acc_clogit_icscore[pos] += 1.0
            elif not logit_correct and not score_correct:
                acc_iclogit_icscore[pos] += 1.0

    acc_clogit_icscore /= num_tries
    acc_iclogit_icscore /= num_tries
    return acc_clogit_icscore, acc_iclogit_icscore


def plot_cinclogits(results_ln0, results_ln1, plot_dir, attn_layer=0, tag=""):
    cl_ic_0, icl_ic_0 = results_ln0
    cl_ic_1, icl_ic_1 = results_ln1

    frac_ic_0 = np.mean(cl_ic_0 + icl_ic_0)
    frac_ic_1 = np.mean(cl_ic_1 + icl_ic_1)

    eps = 1e-10
    corr_0 = np.sum(cl_ic_0) / (np.sum(cl_ic_0 + icl_ic_0) + eps)
    corr_1 = np.sum(cl_ic_1) / (np.sum(cl_ic_1 + icl_ic_1) + eps)

    fig, ax = plt.subplots(figsize=(5, 4.2))
    bw = 0.32
    x = np.array([0, 1])
    b1 = ax.bar(x[0] - bw / 2, frac_ic_0, bw, color='#6a3d9a', label='Without LayerNorm')
    b2 = ax.bar(x[0] + bw / 2, frac_ic_1, bw, color='#e6850e', label='With LayerNorm')
    b3 = ax.bar(x[1] - bw / 2, corr_0, bw, color='#6a3d9a')
    b4 = ax.bar(x[1] + bw / 2, corr_1, bw, color='#e6850e')
    for bar in [b1, b2, b3, b4]:
        for b in bar:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, h + 0.008,
                    f'{h:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Fraction of\nincorrect scores',
                        'Logit correction ratio\namong incorrect scores'], fontsize=11)
    ax.set_ylabel('Fraction', fontsize=12)
    title = f'Incorrect scores & logit correction (Layer {attn_layer})'
    if tag:
        title += f'\n{tag}'
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(True, axis='y', alpha=0.2, linestyle=':')
    ax.tick_params(labelsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ymax = max(frac_ic_0, frac_ic_1, corr_0, corr_1)
    ax.set_ylim(0, ymax * 1.25 if ymax > 0 else 1.0)
    fig.tight_layout()
    path = os.path.join(plot_dir, f'compare_cinclogits_layer{attn_layer}.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {path}")
    print(f"  Frac IC — LN0: {frac_ic_0:.4f}, LN1: {frac_ic_1:.4f}")
    print(f"  Corr    — LN0: {corr_0:.4f}, LN1: {corr_1:.4f}")


# ── Plot 2 & 3: compare_intensity ───────────────────────────────────────────

def compute_intensity(model, config, device, attn_layer, num_rounds=500, fn=200):
    block_size = config.block_size
    vocab_size = config.vocab_size
    location = block_size + 5
    intensity_values = [-0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]

    results = {}
    for intensity in intensity_values:
        attempts = []
        for _ in range(num_rounds):
            if len(attempts) >= fn:
                break
            idx = get_batch(vocab_size, block_size).to(device)
            try:
                im = GPTIntervention(model, idx)
                im.intervent_attention(
                    attention_layer_num=attn_layer, location=location,
                    unsorted_lb=5, unsorted_ub=5,
                    unsorted_lb_num=0, unsorted_ub_num=1,
                    unsorted_intensity_inc=intensity,
                    sorted_lb=0, sorted_num=0, sorted_intensity_inc=0.0,
                )
                new_gen, next_num = im.check_if_still_works()
                attempts.append(new_gen == next_num)
                im.revert_attention(attn_layer)
            except:
                continue
        first = attempts[:fn]
        results[intensity] = sum(first) / len(first) if first else 0
    return intensity_values, results


def plot_intensity(results_ln0, results_ln1, attn_layer, plot_dir, tag=""):
    intensities_0, vals_0 = results_ln0
    intensities_1, vals_1 = results_ln1

    plt.figure(figsize=(3.5, 2.8))
    plt.plot(intensities_0, [vals_0[i] for i in intensities_0],
             marker='o', linewidth=1.5, markersize=5,
             label='without layer norm', color='#1f77b4')
    plt.plot(intensities_1, [vals_1[i] for i in intensities_1],
             marker='s', linewidth=1.5, markersize=5,
             label='with layer norm', color='#ff7f0e')
    plt.xlabel('Intervention Intensity', fontsize=9)
    plt.ylabel('Success Probability', fontsize=9)
    title = f'Robustness to Attention Intervention (Layer {attn_layer})'
    if tag:
        title += f'\n{tag}'
    plt.title(title, fontsize=10)
    plt.legend(fontsize=7, loc='lower left')
    plt.grid(True, alpha=0.3)
    plt.xticks(intensities_0[::2], fontsize=8)
    plt.yticks(fontsize=8)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    path = os.path.join(plot_dir, f'compare_intensity_layer{attn_layer}.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    lr_str = args.lr
    vs, bs, mi = args.vocab_size, args.block_size, args.max_iters
    device = args.device

    run_name_0 = f"V{vs}_B{bs}_LR{lr_str}_MI{mi}_LN0_E64_H1_L2"
    run_name_1 = f"V{vs}_B{bs}_LR{lr_str}_MI{mi}_LN1_E64_H1_L2"
    dir_0 = os.path.join(OUTPUT_BASE, run_name_0)
    dir_1 = os.path.join(OUTPUT_BASE, run_name_1)

    ckpt_0 = find_final_ckpt(dir_0, mi)
    ckpt_1 = find_final_ckpt(dir_1, mi)
    if not ckpt_0:
        print(f"ERROR: Final checkpoint not found in {dir_0}")
        sys.exit(1)
    if not ckpt_1:
        print(f"ERROR: Final checkpoint not found in {dir_1}")
        sys.exit(1)

    model_0, cfg_0, itr_0 = load_model(ckpt_0, device)
    model_1, cfg_1, itr_1 = load_model(ckpt_1, device)

    ckpt_itr = itr_0 if itr_0 == itr_1 else max(itr_0, itr_1)
    plot_dir = os.path.join(OUTPUT_BASE,
                            f"plots_V{vs}_B{bs}_LR{lr_str}_MI{mi}_E64_H1_L2_ckpt{ckpt_itr}")
    os.makedirs(plot_dir, exist_ok=True)

    ckpt_str = f"ckpt={itr_0}" if itr_0 == itr_1 else f"ckpt LN0={itr_0} LN1={itr_1}"
    tag = f"vocab={vs}  block={bs}  lr={lr_str}  iters={mi}  {ckpt_str}"

    print(f"\n{'='*60}")
    print(f"Analyzing: vocab={vs} block={bs} lr={lr_str} iters={mi} {ckpt_str}")
    print(f"  LN0: {ckpt_0}")
    print(f"  LN1: {ckpt_1}")
    print(f"  Output: {plot_dir}")
    print(f"{'='*60}\n")

    # Run all 8 independent computations in parallel via threads
    # (CUDA ops release the GIL so threads give true parallelism)
    results = {}

    def run_cinclogits(model, cfg, label, layer):
        print(f"Computing cinclogits layer {layer} ({label})...")
        return compute_cinclogits(model, cfg, device, attn_layer=layer)

    def run_intensity(model, cfg, label, layer):
        print(f"Computing intensity layer {layer} ({label})...")
        return compute_intensity(model, cfg, device, attn_layer=layer)

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {}
        for layer in [0, 1]:
            futures[pool.submit(run_cinclogits, model_0, cfg_0, "LN0", layer)] = ('cinc', 0, layer)
            futures[pool.submit(run_cinclogits, model_1, cfg_1, "LN1", layer)] = ('cinc', 1, layer)
            futures[pool.submit(run_intensity, model_0, cfg_0, "LN0", layer)] = ('int', 0, layer)
            futures[pool.submit(run_intensity, model_1, cfg_1, "LN1", layer)] = ('int', 1, layer)

        for future in as_completed(futures):
            key = futures[future]
            results[key] = future.result()

    # Assemble and plot
    for layer in [0, 1]:
        plot_cinclogits(results[('cinc', 0, layer)], results[('cinc', 1, layer)],
                        plot_dir, attn_layer=layer, tag=tag)
        plot_intensity(results[('int', 0, layer)], results[('int', 1, layer)],
                       attn_layer=layer, plot_dir=plot_dir, tag=tag)

    print(f"\nAll plots saved to {plot_dir}")


if __name__ == '__main__':
    main()
