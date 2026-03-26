"""
For LN1 V512_B32_LR1e-02 ckpt20000, layer 0:
For each sorted position, check if SEP and/or self-token appear in top-5.
Report exact per-sample and aggregate numbers.
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


def analyze_sample(model, config, device, sample_idx, layer=0):
    block_size = config.block_size
    vocab_size = config.vocab_size
    sep_pos = block_size

    idx = get_batch(vocab_size, block_size).to(device)
    with torch.no_grad():
        model(idx)

    attn = model.transformer.h[layer].c_attn.attn
    seq = idx[0].cpu().tolist()

    sep_in_top5 = 0
    self_in_top5 = 0
    total = 0
    details = []

    for j in range(block_size, 2 * block_size):
        pos_in_sorted = j - block_size
        current_token = seq[j]
        attn_row = attn[j, :j+1].cpu().numpy()
        top5_indices = np.argsort(attn_row)[-5:][::-1]
        top5_tokens = [seq[k] for k in top5_indices]
        top5_scores = attn_row[top5_indices]

        has_sep = sep_pos in top5_indices
        # self = any position in top5 that holds the same token as current position
        # But more precisely: does position j itself appear in top5, OR does another
        # position with the same token value appear?
        # The user likely means: does the query position attend to itself?
        # Position j is in the sorted half. Check if j is in top5_indices.
        has_self_pos = j in top5_indices
        # Also check: does any position in top5 hold the same token value as current?
        has_self_token = current_token in top5_tokens

        if has_sep:
            sep_in_top5 += 1
        if has_self_pos:
            self_in_top5 += 1
        total += 1

        details.append({
            'pos': j,
            'sorted_idx': pos_in_sorted,
            'current': current_token,
            'target': seq[j+1] if j+1 < len(seq) else None,
            'top5_pos': top5_indices.tolist(),
            'top5_tokens': top5_tokens,
            'top5_scores': top5_scores.tolist(),
            'has_sep': has_sep,
            'has_self_pos': has_self_pos,
            'has_self_token': has_self_token,
        })

    return {
        'total': total,
        'sep_in_top5': sep_in_top5,
        'self_pos_in_top5': self_in_top5,
        'self_token_in_top5': sum(1 for d in details if d['has_self_token']),
        'details': details,
    }


def main():
    device = 'cpu'
    output_base = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')

    ckpt_path = os.path.join(
        output_base,
        "V512_B32_LR1e-02_MI20000_LN1_E64_H1_L2",
        "V512_B32_LR1e-02_MI20000_LN1_E64_H1_L2_itr20000.pt"
    )
    model, config = load_model(ckpt_path, device)
    block_size = config.block_size

    print("LN1 (with LayerNorm), V512 B32, ckpt=20000, Layer 0")
    print(f"block_size={block_size}, vocab_size={config.vocab_size}")
    print()

    # Same seed as before
    torch.manual_seed(42)

    all_sep = 0
    all_self_pos = 0
    all_self_token = 0
    all_total = 0

    for i in range(3):
        res = analyze_sample(model, config, device, sample_idx=i+1, layer=0)
        all_sep += res['sep_in_top5']
        all_self_pos += res['self_pos_in_top5']
        all_self_token += res['self_token_in_top5']
        all_total += res['total']

        print(f"{'='*80}")
        print(f"Sample {i+1}: {res['sep_in_top5']}/{res['total']} have SEP in top-5, "
              f"{res['self_pos_in_top5']}/{res['total']} have self-position in top-5, "
              f"{res['self_token_in_top5']}/{res['total']} have self-token-value in top-5")
        print(f"{'='*80}")
        for d in res['details']:
            sep_mark = "SEP" if d['has_sep'] else "   "
            self_mark = "SELF-POS" if d['has_self_pos'] else "        "
            self_tok = "SELF-TOK" if d['has_self_token'] else "        "
            top5_str = ", ".join(
                f"pos={p}(tok={t},attn={s:.4f})"
                for p, t, s in zip(d['top5_pos'], d['top5_tokens'], d['top5_scores'])
            )
            print(f"  sorted[{d['sorted_idx']:2d}] cur={d['current']:3d} tgt={d['target']:3d}  "
                  f"[{sep_mark}] [{self_mark}] [{self_tok}]  top5: {top5_str}")
        print()

    print(f"\n{'='*80}")
    print(f"TOTALS across 3 samples ({all_total} positions):")
    print(f"  SEP in top-5:          {all_sep}/{all_total} = {all_sep/all_total*100:.1f}%")
    print(f"  Self-position in top-5: {all_self_pos}/{all_total} = {all_self_pos/all_total*100:.1f}%")
    print(f"  Self-token in top-5:   {all_self_token}/{all_total} = {all_self_token/all_total*100:.1f}%")
    print(f"{'='*80}")

    # Now large-scale: 500 samples
    print(f"\nLarge-scale: 500 samples...")
    torch.manual_seed(0)
    big_sep = 0
    big_self_pos = 0
    big_self_token = 0
    big_total = 0
    for i in range(500):
        res = analyze_sample(model, config, device, sample_idx=i+1, layer=0)
        big_sep += res['sep_in_top5']
        big_self_pos += res['self_pos_in_top5']
        big_self_token += res['self_token_in_top5']
        big_total += res['total']

    print(f"\n{'='*80}")
    print(f"TOTALS across 500 samples ({big_total} positions):")
    print(f"  SEP in top-5:           {big_sep}/{big_total} = {big_sep/big_total*100:.1f}%")
    print(f"  Self-position in top-5: {big_self_pos}/{big_total} = {big_self_pos/big_total*100:.1f}%")
    print(f"  Self-token in top-5:    {big_self_token}/{big_total} = {big_self_token/big_total*100:.1f}%")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
