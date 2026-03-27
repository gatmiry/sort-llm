"""
Compute how often the SEP token appears in the top-5 attention scores
at layer 0 for sorted-half positions, across many samples.
"""
import sys, os
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_analysis import GPT, GPTConfig


def load_model(ckpt_path, device='cpu'):
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
    return model, config


def get_batch(vocab_size, block_size):
    x = torch.randperm(vocab_size)[:block_size]
    vals, _ = torch.sort(x)
    return torch.cat((x, torch.tensor([vocab_size]), vals), dim=0).unsqueeze(0)


def compute_sep_stats(model, config, device, num_trials=200, layer=0):
    block_size = config.block_size
    vocab_size = config.vocab_size
    sep_pos = block_size  # position of SEP token

    total_positions = 0
    sep_in_top5 = 0
    sep_is_top1 = 0
    # per-position tracking
    per_pos_top5 = np.zeros(block_size)
    per_pos_top1 = np.zeros(block_size)
    # also track sep's average rank and average attention weight
    sep_rank_sum = 0.0
    sep_attn_sum = 0.0

    for _ in range(num_trials):
        idx = get_batch(vocab_size, block_size).to(device)
        with torch.no_grad():
            model(idx)

        attn = model.transformer.h[layer].c_attn.attn

        for j in range(block_size, 2 * block_size):
            pos_in_sorted = j - block_size
            attn_row = attn[j, :j+1].cpu().numpy()
            top5_indices = np.argsort(attn_row)[-5:][::-1]

            total_positions += 1
            sep_attn_val = attn_row[sep_pos]
            sep_attn_sum += sep_attn_val

            # rank of SEP (1-indexed)
            sorted_desc = np.argsort(attn_row)[::-1]
            sep_rank = int(np.where(sorted_desc == sep_pos)[0][0]) + 1
            sep_rank_sum += sep_rank

            if sep_pos in top5_indices:
                sep_in_top5 += 1
                per_pos_top5[pos_in_sorted] += 1
            if top5_indices[0] == sep_pos:
                sep_is_top1 += 1
                per_pos_top1[pos_in_sorted] += 1

    per_pos_top5 /= num_trials
    per_pos_top1 /= num_trials

    return {
        'total': total_positions,
        'sep_in_top5': sep_in_top5,
        'sep_in_top5_pct': sep_in_top5 / total_positions * 100,
        'sep_is_top1': sep_is_top1,
        'sep_is_top1_pct': sep_is_top1 / total_positions * 100,
        'avg_sep_rank': sep_rank_sum / total_positions,
        'avg_sep_attn': sep_attn_sum / total_positions,
        'per_pos_top5': per_pos_top5,
        'per_pos_top1': per_pos_top1,
    }


def main():
    device = 'cpu'
    output_base = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
    num_trials = 200

    torch.manual_seed(0)

    for ln_val, ln_label in [(0, "LN0 (without LayerNorm)"), (1, "LN1 (with LayerNorm)")]:
        ckpt_path = os.path.join(
            output_base,
            f"V512_B32_LR1e-02_MI20000_LN{ln_val}_E64_H1_L2",
            f"V512_B32_LR1e-02_MI20000_LN{ln_val}_E64_H1_L2_itr20000.pt"
        )
        print(f"\n{'='*65}")
        print(f"  {ln_label}  (V512, B32, LR=1e-02, ckpt=20000)")
        print(f"  Layer 0, {num_trials} trials x 32 positions = {num_trials*32} total")
        print(f"{'='*65}")

        model, config = load_model(ckpt_path, device)
        res = compute_sep_stats(model, config, device, num_trials=num_trials, layer=0)

        print(f"\n  SEP in top-5:  {res['sep_in_top5']:5d}/{res['total']}  = {res['sep_in_top5_pct']:.1f}%")
        print(f"  SEP is #1:     {res['sep_is_top1']:5d}/{res['total']}  = {res['sep_is_top1_pct']:.1f}%")
        print(f"  Avg SEP rank:  {res['avg_sep_rank']:.2f}")
        print(f"  Avg SEP attn:  {res['avg_sep_attn']:.4f}")

        print(f"\n  Per-position breakdown (fraction of trials where SEP is in top-5 / is #1):")
        print(f"  {'pos':>4s}  {'top5':>6s}  {'#1':>6s}  {'top5 bar'}")
        for i in range(config.block_size):
            bar5 = '#' * int(res['per_pos_top5'][i] * 50)
            print(f"  {i:4d}  {res['per_pos_top5'][i]:6.3f}  {res['per_pos_top1'][i]:6.3f}  {bar5}")


if __name__ == '__main__':
    main()
