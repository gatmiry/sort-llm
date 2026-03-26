#!/usr/bin/env python3
"""
1.5M-iteration training for sort-GPT.
Matches 200k-checkpoints config exactly (same architecture, LR, batch size).
Saves checkpoints every 50k iterations + final.

Usage: python train.py --gpu 0 --init-seed 1337
"""
import argparse
import math
import os
import sys
import time
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from model_tbyt_train import GPT, GPTConfig

VOCAB_SIZE = 256
BLOCK_SIZE = 16
N_LAYERS = 2
N_HEADS = 1
N_EMBD = 64

MAX_ITERS = 1_500_000
CKPT_INTERVAL = 50_000
BATCH_SIZE = 4096
WARMUP_ITERS = 200
LEARNING_RATE = 0.03
MIN_LR = 1e-6
WEIGHT_DECAY = 0.0
DATA_SEED = 1337
LOG_INTERVAL = 5000


def get_lr(it):
    if it < WARMUP_ITERS:
        return LEARNING_RATE * (it + 1) / (WARMUP_ITERS + 1)
    if it >= MAX_ITERS:
        return MIN_LR
    ratio = (it - WARMUP_ITERS) / (MAX_ITERS - WARMUP_ITERS)
    return MIN_LR + 0.5 * (1.0 + math.cos(math.pi * ratio)) * (LEARNING_RATE - MIN_LR)


def get_batch(device):
    """Vectorized batch generation on GPU — no Python loops."""
    ids = torch.rand(BATCH_SIZE, VOCAB_SIZE, device=device).argsort(dim=1)[:, :BLOCK_SIZE]
    sorted_ids, _ = ids.sort(dim=1)
    sep = torch.full((BATCH_SIZE, 1), VOCAB_SIZE, dtype=torch.long, device=device)
    return torch.cat([ids, sep, sorted_ids], dim=1)


def save_checkpoint(model, iteration, init_seed, save_dir, is_final=False):
    raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    tag = 'final' if is_final else f'ckpt{iteration}'
    filename = (f"sortgpt_k{BLOCK_SIZE}_methfixed_mlp1_L{N_LAYERS}_N{VOCAB_SIZE}_E{N_EMBD}"
                f"_pos0_fln1_wd0p0_lr0p03_dseed{DATA_SEED}_iseed{init_seed}__{tag}.pt")
    path = os.path.join(save_dir, filename)
    torch.save({
        'model_state_dict': raw_model.state_dict(),
        'model_config': {
            'block_size': BLOCK_SIZE, 'vocab_size': VOCAB_SIZE + 1,
            'n_layers': N_LAYERS, 'n_heads': N_HEADS, 'n_embd': N_EMBD,
            'without_pos': True, 'use_mlp': True, 'use_final_LN': True,
            'max_seq_len': 193,
        },
        'train_config': {
            'max_iters': MAX_ITERS, 'effective_batch_size': BATCH_SIZE,
            'warmup_iters': WARMUP_ITERS, 'learning_rate': LEARNING_RATE,
            'min_lr': MIN_LR, 'weight_decay': WEIGHT_DECAY,
            'data_seed': DATA_SEED, 'init_seed': init_seed,
            'train_length_method': 'fixed', 'allow_duplicates': False,
        },
        'iteration': iteration,
    }, path)
    return filename


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--init-seed', type=int, required=True)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = 'cuda'
    seed = args.init_seed

    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'final_models')
    os.makedirs(save_dir, exist_ok=True)

    final_name = (f"sortgpt_k{BLOCK_SIZE}_methfixed_mlp1_L{N_LAYERS}_N{VOCAB_SIZE}_E{N_EMBD}"
                  f"_pos0_fln1_wd0p0_lr0p03_dseed{DATA_SEED}_iseed{seed}__final.pt")
    if os.path.exists(os.path.join(save_dir, final_name)):
        print(f"[seed={seed}] Final checkpoint exists, skipping.", flush=True)
        return

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.set_float32_matmul_precision('high')

    config = GPTConfig(block_size=BLOCK_SIZE, vocab_size=VOCAB_SIZE)
    model = GPT(config).to(device)
    model = torch.compile(model)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE,
        betas=(0.9, 0.95), eps=1e-8, weight_decay=WEIGHT_DECAY, fused=True)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[seed={seed}] GPU {args.gpu} | {n_params:,} params | "
          f"{MAX_ITERS:,} iters | batch={BATCH_SIZE}", flush=True)

    t_start = time.time()
    for it in range(MAX_ITERS):
        lr = get_lr(it)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        _, loss = model(get_batch(device))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if it % LOG_INTERVAL == 0:
            elapsed = time.time() - t_start
            ms_per_it = elapsed / max(it, 1) * 1000
            eta_min = (MAX_ITERS - it) * ms_per_it / 60_000
            print(f"[seed={seed}] it={it:>8d}  loss={loss.item():.3e}  "
                  f"lr={lr:.5f}  {ms_per_it:.1f}ms/it  ETA~{eta_min:.0f}m",
                  flush=True)

        if (it + 1) % CKPT_INTERVAL == 0:
            fname = save_checkpoint(model, it + 1, seed, save_dir)
            print(f"[seed={seed}] Saved {fname}", flush=True)

    fname = save_checkpoint(model, MAX_ITERS, seed, save_dir, is_final=True)
    total_min = (time.time() - t_start) / 60
    print(f"[seed={seed}] DONE in {total_min:.1f}min — {fname}", flush=True)


if __name__ == '__main__':
    main()
