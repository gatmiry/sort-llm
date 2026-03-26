"""
For both LN0 and LN1 V512_B32_LR1e-02 ckpt20000, layer 0:
Check if SEP and self appear in top-5 and top-10.
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


def compute_stats(model, config, device, num_trials=500, layer=0):
    block_size = config.block_size
    sep_pos = block_size

    counts = {k: 0 for k in [
        'sep_top5', 'sep_top10',
        'self_pos_top5', 'self_pos_top10',
        'self_tok_top5', 'self_tok_top10',
    ]}
    total = 0

    for _ in range(num_trials):
        idx = get_batch(config.vocab_size, block_size).to(device)
        with torch.no_grad():
            model(idx)

        attn = model.transformer.h[layer].c_attn.attn
        seq = idx[0].cpu().tolist()

        for j in range(block_size, 2 * block_size):
            attn_row = attn[j, :j+1].cpu().numpy()
            sorted_desc = np.argsort(attn_row)[::-1]
            top5 = set(sorted_desc[:5].tolist())
            top10 = set(sorted_desc[:10].tolist())
            top5_tokens = {seq[k] for k in top5}
            top10_tokens = {seq[k] for k in top10}
            cur_tok = seq[j]

            total += 1
            if sep_pos in top5:
                counts['sep_top5'] += 1
            if sep_pos in top10:
                counts['sep_top10'] += 1
            if j in top5:
                counts['self_pos_top5'] += 1
            if j in top10:
                counts['self_pos_top10'] += 1
            if cur_tok in top5_tokens:
                counts['self_tok_top5'] += 1
            if cur_tok in top10_tokens:
                counts['self_tok_top10'] += 1

    return counts, total


def main():
    device = 'cpu'
    output_base = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
    num_trials = 500

    for ln_val, ln_label in [(0, "LN0 (without LayerNorm)"), (1, "LN1 (with LayerNorm)")]:
        ckpt_path = os.path.join(
            output_base,
            f"V512_B32_LR1e-02_MI20000_LN{ln_val}_E64_H1_L2",
            f"V512_B32_LR1e-02_MI20000_LN{ln_val}_E64_H1_L2_itr20000.pt"
        )
        model, config = load_model(ckpt_path, device)

        torch.manual_seed(0)
        counts, total = compute_stats(model, config, device, num_trials=num_trials, layer=0)

        print(f"\n{'='*65}")
        print(f"  {ln_label}  (V512, B32, ckpt=20000, Layer 0)")
        print(f"  {num_trials} trials x 32 positions = {total} total")
        print(f"{'='*65}")
        print(f"                        {'top-5':>12s}  {'top-10':>12s}")
        print(f"  SEP in top-K:         {counts['sep_top5']:5d}/{total} ({counts['sep_top5']/total*100:5.1f}%)  "
              f"{counts['sep_top10']:5d}/{total} ({counts['sep_top10']/total*100:5.1f}%)")
        print(f"  Self-position in top-K:{counts['self_pos_top5']:5d}/{total} ({counts['self_pos_top5']/total*100:5.1f}%)  "
              f"{counts['self_pos_top10']:5d}/{total} ({counts['self_pos_top10']/total*100:5.1f}%)")
        print(f"  Self-token in top-K:  {counts['self_tok_top5']:5d}/{total} ({counts['self_tok_top5']/total*100:5.1f}%)  "
              f"{counts['self_tok_top10']:5d}/{total} ({counts['self_tok_top10']/total*100:5.1f}%)")


if __name__ == '__main__':
    main()
