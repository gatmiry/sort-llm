"""
Verify the cinclogits computation: for each sorted position, check whether
the argmax of layer-0 attention points to the correct next token.
Shows per-position breakdown and the aggregate fraction.
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


def compute_cinclogits_verbose(model, config, device, num_tries=100, attn_layer=0):
    block_size = config.block_size
    vocab_size = config.vocab_size
    acc_clogit_icscore = np.zeros(block_size)
    acc_iclogit_icscore = np.zeros(block_size)
    total_incorrect_score = 0
    total_positions = 0

    # Track what the argmax points to when incorrect
    argmax_is_sep = 0
    argmax_is_unsorted_other = 0
    argmax_is_sorted = 0
    total_incorrect = 0

    for trial in range(num_tries):
        idx = get_batch(vocab_size, block_size).to(device)
        with torch.no_grad():
            logits, _ = model(idx)

        is_correct = (torch.argmax(logits[0, block_size:2 * block_size, :], dim=1)
                      == idx[0, block_size + 1:])
        attn_weights = model.transformer.h[attn_layer].c_attn.attn

        for j in range(block_size, 2 * block_size):
            max_score = float('-inf')
            max_score_pos = -1
            max_score_num = -1
            for k in range(0, 2 * block_size + 1):
                score = attn_weights[j, k].item()
                if score > max_score:
                    max_score = score
                    max_score_pos = k
                    max_score_num = idx[0, k].item()

            target_token = idx[0, j + 1].item()
            score_correct = (max_score_num == target_token)
            pos = j - block_size
            logit_correct = is_correct[pos].item()

            total_positions += 1
            if not score_correct:
                total_incorrect += 1
                if max_score_pos == block_size:
                    argmax_is_sep += 1
                elif max_score_pos < block_size:
                    argmax_is_unsorted_other += 1
                else:
                    argmax_is_sorted += 1

            if logit_correct and not score_correct:
                acc_clogit_icscore[pos] += 1.0
            elif not logit_correct and not score_correct:
                acc_iclogit_icscore[pos] += 1.0

        if trial < 3:
            seq = idx[0].cpu().tolist()
            print(f"\n  Trial {trial+1} detail (layer {attn_layer}):")
            incorrect_positions = []
            for j in range(block_size, 2 * block_size):
                max_k = -1
                max_s = float('-inf')
                for k in range(0, 2 * block_size + 1):
                    s = attn_weights[j, k].item()
                    if s > max_s:
                        max_s = s
                        max_k = k
                argmax_token = seq[max_k]
                target = seq[j + 1]
                ok = "ok" if argmax_token == target else "WRONG"
                if ok == "WRONG":
                    incorrect_positions.append(j - block_size)
                region = "unsorted" if max_k < block_size else ("SEP" if max_k == block_size else "sorted")
                print(f"    sorted[{j-block_size:2d}]: argmax -> pos {max_k:2d} ({region:>8s}) "
                      f"token={argmax_token:3d} (attn={max_s:.4f}), target={target:3d} [{ok}]")
            print(f"    Incorrect: {len(incorrect_positions)}/{block_size} "
                  f"at positions {incorrect_positions}")

    acc_clogit_icscore /= num_tries
    acc_iclogit_icscore /= num_tries

    frac_ic = np.mean(acc_clogit_icscore + acc_iclogit_icscore)

    return {
        'frac_ic': frac_ic,
        'per_pos_ic': acc_clogit_icscore + acc_iclogit_icscore,
        'total_incorrect': total_incorrect,
        'total_positions': total_positions,
        'argmax_is_sep': argmax_is_sep,
        'argmax_is_unsorted_other': argmax_is_unsorted_other,
        'argmax_is_sorted': argmax_is_sorted,
    }


def main():
    device = 'cpu'
    output_base = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
    torch.manual_seed(0)

    for ln_label, ln_val in [("LN0 (without LayerNorm)", 0), ("LN1 (with LayerNorm)", 1)]:
        ckpt_path = os.path.join(
            output_base,
            f"V256_B16_LR1e-02_MI60000_LN{ln_val}_E64_H1_L2",
            f"V256_B16_LR1e-02_MI60000_LN{ln_val}_E64_H1_L2_itr60000.pt"
        )
        print(f"\n{'='*70}")
        print(f"  {ln_label}  — Layer 0")
        print(f"{'='*70}")

        model, config = load_model(ckpt_path, device)
        res = compute_cinclogits_verbose(model, config, device, num_tries=100, attn_layer=0)

        print(f"\n  AGGREGATE RESULTS (100 trials × 16 positions = {res['total_positions']} total):")
        print(f"  Fraction of incorrect scores: {res['frac_ic']:.4f}")
        print(f"  Total incorrect: {res['total_incorrect']}/{res['total_positions']} "
              f"= {res['total_incorrect']/res['total_positions']:.4f}")
        print(f"\n  When argmax is WRONG, it points to:")
        print(f"    SEP token:           {res['argmax_is_sep']:4d} "
              f"({res['argmax_is_sep']/max(res['total_incorrect'],1)*100:.1f}%)")
        print(f"    Other unsorted token: {res['argmax_is_unsorted_other']:4d} "
              f"({res['argmax_is_unsorted_other']/max(res['total_incorrect'],1)*100:.1f}%)")
        print(f"    Sorted-half token:   {res['argmax_is_sorted']:4d} "
              f"({res['argmax_is_sorted']/max(res['total_incorrect'],1)*100:.1f}%)")

        print(f"\n  Per-position incorrect-score rate:")
        for i, v in enumerate(res['per_pos_ic']):
            bar = '#' * int(v * 50)
            print(f"    sorted[{i:2d}]: {v:.3f}  {bar}")


if __name__ == '__main__':
    main()
