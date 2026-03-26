"""
Fine-grained analysis worker. Computes ONE metric for ONE checkpoint on ONE layer.
Saves result to a numpy file. The launcher collects results and generates plots.

Usage:
  python analyze_worker.py --ckpt PATH --task cinclogits --layer 0 --out result.npz
  python analyze_worker.py --ckpt PATH --task intensity --layer 1 --out result.npz
"""
import argparse
import os
import numpy as np
import torch
from model_analysis import GPT, GPTConfig, GPTIntervention


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', type=str, required=True)
    p.add_argument('--task', type=str, required=True, choices=['cinclogits', 'intensity', 'ablation', 'baseline'])
    p.add_argument('--layer', type=int, default=0)
    p.add_argument('--out', type=str, required=True)
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--unsorted_lb', type=int, default=5)
    p.add_argument('--unsorted_ub', type=int, default=5)
    return p.parse_args()


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


def compute_cinclogits(model, config, device, attn_layer, num_tries=100):
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


def compute_intensity(model, config, device, attn_layer, unsorted_lb=5, unsorted_ub=5, min_valid=200):
    block_size = config.block_size
    vocab_size = config.vocab_size
    location = block_size + 5
    intensity_values = [-0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5,
                        1.75, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0]

    success_rates = []
    counts = []
    for intensity in intensity_values:
        attempts = []
        rounds = 0
        while len(attempts) < min_valid and rounds < 2000:
            rounds += 1
            idx = get_batch(vocab_size, block_size).to(device)
            try:
                im = GPTIntervention(model, idx)
                im.intervent_attention(
                    attention_layer_num=attn_layer, location=location,
                    unsorted_lb=unsorted_lb, unsorted_ub=unsorted_ub,
                    unsorted_lb_num=0, unsorted_ub_num=1,
                    unsorted_intensity_inc=intensity,
                    sorted_lb=0, sorted_num=0, sorted_intensity_inc=0.0,
                )
                new_gen, next_num = im.check_if_still_works()
                attempts.append(new_gen == next_num)
                im.revert_attention(attn_layer)
            except:
                continue
        n = len(attempts)
        counts.append(n)
        if n < min_valid:
            print(f"  WARNING: intensity={intensity:.2f} got {n}/{min_valid} valid after 2000 rounds", flush=True)
        success_rates.append(sum(attempts) / n if n > 0 else 0.0)
    print(f"  Intensity counts per value: {dict(zip(intensity_values, counts))}", flush=True)
    return np.array(intensity_values), np.array(success_rates), np.array(counts)


def compute_ablation(model, config, device, skip_layer, num_trials=500):
    """Test accuracy when bypassing attention+layernorm for a specific layer."""
    block_size = config.block_size
    vocab_size = config.vocab_size

    block = model.transformer.h[skip_layer]
    original_forward = block.forward

    def forward_skip_attn(x, layer_n=-1):
        return x + block.c_fc(block.ln_2(x))

    block.forward = forward_skip_attn

    per_pos_correct = np.zeros(block_size)
    full_seq_correct = 0
    cond_correct = np.zeros(block_size)
    cond_eligible = np.zeros(block_size)

    try:
        for _ in range(num_trials):
            idx = get_batch(vocab_size, block_size).to(device)
            with torch.no_grad():
                logits, _ = model(idx)
            preds = torch.argmax(logits[0, block_size:2 * block_size, :], dim=1)
            targets = idx[0, block_size + 1:]
            correct = (preds == targets).cpu().numpy()
            per_pos_correct += correct
            if correct.all():
                full_seq_correct += 1
            prefix_correct = True
            for i in range(block_size):
                if prefix_correct:
                    cond_eligible[i] += 1
                    if correct[i]:
                        cond_correct[i] += 1
                    else:
                        prefix_correct = False
                else:
                    break
    finally:
        block.forward = original_forward

    per_pos_acc = per_pos_correct / num_trials
    full_seq_acc = full_seq_correct / num_trials
    cond_acc = np.where(cond_eligible > 0, cond_correct / cond_eligible, 0.0)
    print(f"  skip_layer={skip_layer}: full_seq_acc={full_seq_acc:.4f}, "
          f"mean_pos_acc={per_pos_acc.mean():.4f}, "
          f"mean_cond_acc={cond_acc[cond_eligible > 0].mean():.4f}", flush=True)
    return per_pos_acc, full_seq_acc, cond_acc, cond_eligible


def compute_baseline(model, config, device, num_trials=500):
    """Test accuracy of the intact model with teacher forcing."""
    block_size = config.block_size
    vocab_size = config.vocab_size

    per_pos_correct = np.zeros(block_size)
    full_seq_correct = 0
    cond_correct = np.zeros(block_size)
    cond_eligible = np.zeros(block_size)

    for _ in range(num_trials):
        idx = get_batch(vocab_size, block_size).to(device)
        with torch.no_grad():
            logits, _ = model(idx)
        preds = torch.argmax(logits[0, block_size:2 * block_size, :], dim=1)
        targets = idx[0, block_size + 1:]
        correct = (preds == targets).cpu().numpy()
        per_pos_correct += correct
        if correct.all():
            full_seq_correct += 1
        prefix_correct = True
        for i in range(block_size):
            if prefix_correct:
                cond_eligible[i] += 1
                if correct[i]:
                    cond_correct[i] += 1
                else:
                    prefix_correct = False
            else:
                break

    per_pos_acc = per_pos_correct / num_trials
    full_seq_acc = full_seq_correct / num_trials
    cond_acc = np.where(cond_eligible > 0, cond_correct / cond_eligible, 0.0)
    print(f"  baseline: full_seq_acc={full_seq_acc:.4f}, "
          f"mean_pos_acc={per_pos_acc.mean():.4f}, "
          f"mean_cond_acc={cond_acc[cond_eligible > 0].mean():.4f}", flush=True)
    return per_pos_acc, full_seq_acc, cond_acc, cond_eligible


def main():
    args = parse_args()
    model, config, itr = load_model(args.ckpt, args.device)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    if args.task == 'cinclogits':
        cl_ic, icl_ic = compute_cinclogits(model, config, args.device, args.layer)
        np.savez(args.out, clogit_icscore=cl_ic, iclogit_icscore=icl_ic, itr=itr)
    elif args.task == 'intensity':
        intensities, rates, counts = compute_intensity(
            model, config, args.device, args.layer,
            unsorted_lb=args.unsorted_lb, unsorted_ub=args.unsorted_ub)
        np.savez(args.out, intensities=intensities, success_rates=rates, counts=counts, itr=itr)
    elif args.task == 'ablation':
        per_pos_acc, full_seq_acc, cond_acc, cond_eligible = compute_ablation(
            model, config, args.device, skip_layer=args.layer)
        np.savez(args.out, per_pos_acc=per_pos_acc, full_seq_acc=full_seq_acc,
                 cond_acc=cond_acc, cond_eligible=cond_eligible,
                 skip_layer=args.layer, itr=itr)
    elif args.task == 'baseline':
        per_pos_acc, full_seq_acc, cond_acc, cond_eligible = compute_baseline(
            model, config, args.device)
        np.savez(args.out, per_pos_acc=per_pos_acc, full_seq_acc=full_seq_acc,
                 cond_acc=cond_acc, cond_eligible=cond_eligible, itr=itr)

    print(f"Saved {args.out}")


if __name__ == '__main__':
    main()
