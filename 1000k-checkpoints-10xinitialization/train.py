"""
Training script for 1M-iteration run with 10x initialization std.
Identical to 1000k-checkpoints config except init std = 0.2 (vs 0.02).
Saves checkpoints every 50k iterations + final model.

Usage:
  python train.py --gpu 0
"""
import argparse
import math
import os
import sys
import time
import json
import torch
from model_tbyt_train import GPT, GPTConfig


VOCAB_SIZE = 256
BLOCK_SIZE = 16
N_LAYERS = 2
N_HEADS = 1
N_EMBD = 64
MAX_SEQ_LEN = 193

MAX_ITERS = 1000000
CKPT_INTERVAL = 50000
BATCH_SIZE = 4096
MICRO_BATCH = 1024
ACCUM_STEPS = BATCH_SIZE // MICRO_BATCH  # 4
WARMUP_ITERS = 200
LEARNING_RATE = 0.03
MIN_LR = 1e-6
WEIGHT_DECAY = 0.0
DATA_SEED = 1337
INIT_SEED = 1337
WITH_LN = True
LOG_INTERVAL = 1000


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--resume', type=str, default=None,
                   help='Path to checkpoint to resume from')
    return p.parse_args()


def get_lr(itr):
    if itr < WARMUP_ITERS:
        return LEARNING_RATE * (itr + 1) / (WARMUP_ITERS + 1)
    if itr > MAX_ITERS:
        return MIN_LR
    ratio = (itr - WARMUP_ITERS) / (MAX_ITERS - WARMUP_ITERS)
    ratio = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return MIN_LR + ratio * (LEARNING_RATE - MIN_LR)


def save_checkpoint(model, optimizer, config, itr, loss, out_dir, is_final=False):
    model_config = {
        'block_size': BLOCK_SIZE, 'vocab_size': VOCAB_SIZE + 1,
        'n_layers': N_LAYERS, 'n_heads': N_HEADS, 'n_embd': N_EMBD,
        'without_pos': True, 'use_mlp': True,
        'use_final_LN': WITH_LN, 'max_seq_len': MAX_SEQ_LEN,
    }
    train_config = {
        'block_size': BLOCK_SIZE, 'vocab_n': VOCAB_SIZE,
        'n_layers': N_LAYERS, 'n_heads': N_HEADS, 'n_embd': N_EMBD,
        'max_iters': MAX_ITERS, 'effective_batch_size': BATCH_SIZE,
        'warmup_iters': WARMUP_ITERS, 'learning_rate': LEARNING_RATE,
        'min_lr': MIN_LR, 'weight_decay': WEIGHT_DECAY,
        'data_seed': DATA_SEED, 'init_seed': INIT_SEED,
        'use_final_LN': WITH_LN,
        'init_std': 0.2,
        'init_std_note': '10x default (0.02 -> 0.2)',
    }

    tag = f"sortgpt_k{BLOCK_SIZE}_methfixed_mlp1_L{N_LAYERS}_N{VOCAB_SIZE}_E{N_EMBD}_pos0_fln{int(WITH_LN)}_wd0p0_lr0p03_dseed{DATA_SEED}_iseed{INIT_SEED}"
    if is_final:
        name = f"{tag}__final.pt"
    else:
        name = f"{tag}__ckpt{itr}.pt"

    sd = {}
    for k, v in model.state_dict().items():
        clean_k = k.replace('_orig_mod.', '')
        sd[clean_k] = v

    ckpt = {
        'model_state_dict': sd,
        'optimizer_state_dict': optimizer.state_dict(),
        'model_config': model_config,
        'train_config': train_config,
        'iteration': itr,
        'train_loss': loss,
        'artifact_type': 'final_model' if is_final else f'ckpt{itr}',
    }
    path = os.path.join(out_dir, name)
    torch.save(ckpt, path)
    return path


def main():
    args = parse_args()
    device = f'cuda:{args.gpu}'
    torch.cuda.set_device(args.gpu)

    out_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)

    def get_batch(bs):
        scores = torch.rand(bs, VOCAB_SIZE, device=device)
        x = scores.topk(BLOCK_SIZE, dim=1).indices.to(torch.long)
        vals = x.sort(dim=1).values
        sep = torch.full((bs, 1), VOCAB_SIZE, dtype=torch.long, device=device)
        return torch.cat([x, sep, vals], dim=1)

    torch.set_float32_matmul_precision('high')
    torch.manual_seed(INIT_SEED)
    torch.cuda.manual_seed(INIT_SEED)

    config = GPTConfig(block_size=BLOCK_SIZE, vocab_size=VOCAB_SIZE,
                       with_layer_norm=WITH_LN, max_seq_len=MAX_SEQ_LEN)
    model = GPT(config)
    model.to(device)
    model = torch.compile(model)

    params = [p for p in model.parameters() if p.requires_grad]
    decay_params = [p for p in params if p.dim() > 1]
    nondecay_params = [p for p in params if p.dim() <= 1]
    optimizer = torch.optim.AdamW([
        {'params': decay_params, 'weight_decay': WEIGHT_DECAY},
        {'params': nondecay_params, 'weight_decay': 0.0}
    ], lr=LEARNING_RATE, betas=(0.9, 0.95), eps=1e-8)

    start_itr = 0
    if args.resume:
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_itr = ckpt.get('iteration', 0)
        print(f"  Resumed at iteration {start_itr}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Training (10x init std): N={VOCAB_SIZE}, B={BLOCK_SIZE}, lr={LEARNING_RATE}, "
          f"max_iters={MAX_ITERS}, ckpt_interval={CKPT_INTERVAL}")
    print(f"  batch={BATCH_SIZE}, micro={MICRO_BATCH}, accum={ACCUM_STEPS}, "
          f"params={total_params:,}")
    print(f"  init_std=0.2 (10x default)")
    print(f"  output_dir={out_dir}")
    print(f"  GPU: {torch.cuda.get_device_name(args.gpu)}")
    sys.stdout.flush()

    t0 = time.time()
    best_loss = float('inf')
    history = []

    for itr in range(start_itr, MAX_ITERS):
        model.train()
        optimizer.zero_grad()

        for astep in range(ACCUM_STEPS):
            x = get_batch(MICRO_BATCH)
            logits, loss = model(x)
            (loss / ACCUM_STEPS).backward()

        lr = get_lr(itr)
        for pg in optimizer.param_groups:
            pg['lr'] = lr
        optimizer.step()

        if itr % LOG_INTERVAL == 0:
            model.eval()
            with torch.no_grad():
                x_test = get_batch(512)
                _, test_loss = model(x_test)

            train_loss_val = loss.item()
            test_loss_val = test_loss.item()
            elapsed = time.time() - t0
            iters_per_sec = (itr - start_itr + 1) / elapsed if elapsed > 0 else 0
            eta_s = (MAX_ITERS - itr) / iters_per_sec if iters_per_sec > 0 else 0

            print(f"  itr {itr:>7d}/{MAX_ITERS} | loss {train_loss_val:.6f} | "
                  f"test {test_loss_val:.6f} | lr {lr:.2e} | "
                  f"{iters_per_sec:.0f} it/s | eta {eta_s/60:.0f}m | "
                  f"{elapsed/60:.1f}m elapsed", flush=True)

            if itr > 0:
                history.append({
                    'iter': itr, 'lr': lr,
                    'loss': train_loss_val, 'test_loss': test_loss_val,
                })

        if (itr + 1) % CKPT_INTERVAL == 0:
            path = save_checkpoint(model, optimizer, config, itr + 1, loss.item(), out_dir)
            print(f"  [CKPT] Saved {os.path.basename(path)} ({(time.time()-t0)/60:.1f}m)", flush=True)

    path = save_checkpoint(model, optimizer, config, MAX_ITERS, loss.item(), out_dir, is_final=True)
    print(f"  [FINAL] Saved {os.path.basename(path)}")

    elapsed = time.time() - t0
    print(f"\nFinished {MAX_ITERS} iterations in {elapsed/60:.1f}m ({elapsed/3600:.2f}h)")

    hist_path = os.path.join(out_dir, 'training_history.json')
    with open(hist_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"  History saved to {hist_path}")


if __name__ == '__main__':
    main()
