"""
Training script for new-run configs.
Matches the 100k-checkpoints architecture (with_layer_norm=True, weight_decay=0.0).

Usage:
  python train.py --vocab_size 256 --block_size 16 --learning_rate 0.03 \
                  --max_iters 200000 --ckpt_interval 20000 \
                  --data_seed 1337 --init_seed 1338 --gpu 0
"""
import argparse
import math
import os
import time
import torch
from model_tbyt_train import GPT, GPTConfig


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--vocab_size', type=int, required=True)
    p.add_argument('--block_size', type=int, required=True)
    p.add_argument('--max_iters', type=int, required=True)
    p.add_argument('--learning_rate', type=float, required=True)
    p.add_argument('--with_layer_norm', type=int, default=1, choices=[0, 1])
    p.add_argument('--data_seed', type=int, default=1337)
    p.add_argument('--init_seed', type=int, default=1338)
    p.add_argument('--ckpt_interval', type=int, default=20000)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--output_dir', type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()

    device = f'cuda:{args.gpu}'
    torch.cuda.set_device(args.gpu)

    vocab_size = args.vocab_size
    block_size = args.block_size
    max_iters = args.max_iters
    learning_rate = args.learning_rate
    with_ln = bool(args.with_layer_norm)
    data_seed = args.data_seed
    init_seed = args.init_seed
    ckpt_interval = args.ckpt_interval

    batch_size = 4096
    n_layers = 2
    n_heads = 1
    n_embd = 64
    warmup_iters = 200
    min_lr = 1e-6
    weight_decay = 0.0

    micro_batch = batch_size
    accum_steps = 1
    if vocab_size >= 8192:
        micro_batch = 256
        accum_steps = batch_size // micro_batch
    elif vocab_size >= 512:
        micro_batch = 1024
        accum_steps = batch_size // micro_batch

    lr_str = f"{learning_rate}"
    ln_tag = "LN1" if with_ln else "LN0"
    run_tag = f"N{vocab_size}_B{block_size}_lr{lr_str}_ds{data_seed}_is{init_seed}_{ln_tag}"

    if args.output_dir:
        out_dir = args.output_dir
    else:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), run_tag)
    os.makedirs(out_dir, exist_ok=True)

    def get_batch(bs):
        scores = torch.rand(bs, vocab_size, device=device)
        x = scores.topk(block_size, dim=1).indices.to(torch.long)
        vals = x.sort(dim=1).values
        sep = torch.full((bs, 1), vocab_size, dtype=torch.long, device=device)
        return torch.cat([x, sep, vals], dim=1)

    def get_lr(itr):
        if itr < warmup_iters:
            return learning_rate * (itr + 1) / (warmup_iters + 1)
        if itr > max_iters:
            return min_lr
        ratio = (itr - warmup_iters) / (max_iters - warmup_iters)
        ratio = 0.5 * (1.0 + math.cos(math.pi * ratio))
        return min_lr + ratio * (learning_rate - min_lr)

    torch.set_float32_matmul_precision('high')
    torch.manual_seed(init_seed)
    torch.cuda.manual_seed(init_seed)
    config = GPTConfig(block_size=block_size, vocab_size=vocab_size, with_layer_norm=with_ln)
    model = GPT(config)
    model.to(device)
    model = torch.compile(model)

    params = [p for p in model.parameters() if p.requires_grad]
    decay_params = [p for p in params if p.dim() > 1]
    nondecay_params = [p for p in params if p.dim() <= 1]
    optimizer = torch.optim.AdamW([
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nondecay_params, 'weight_decay': 0.0}
    ], lr=learning_rate, betas=(0.9, 0.95), eps=1e-8)

    print(f"Starting: {run_tag}")
    print(f"  max_iters={max_iters}, lr={learning_rate}, ckpt_interval={ckpt_interval}")
    print(f"  batch_size={batch_size}, micro_batch={micro_batch}, accum_steps={accum_steps}")
    print(f"  output_dir={out_dir}")
    t0 = time.time()

    for itr in range(max_iters):
        model.train()
        optimizer.zero_grad()

        for astep in range(accum_steps):
            x = get_batch(micro_batch)
            logits, loss = model(x)
            (loss / accum_steps).backward()

        lr = get_lr(itr)
        for pg in optimizer.param_groups:
            pg['lr'] = lr
        optimizer.step()

        if itr % 1000 == 0:
            model.eval()
            with torch.no_grad():
                x_test = get_batch(min(micro_batch, 512))
                _, test_loss = model(x_test)
            elapsed = time.time() - t0
            print(f"  itr {itr}/{max_iters} | train_loss {loss.item():.4f} | test_loss {test_loss.item():.4f} | lr {lr:.2e} | {elapsed:.0f}s", flush=True)

        if (itr + 1) % ckpt_interval == 0 or itr == max_iters - 1:
            ckpt_name = f"{run_tag}_ckpt{itr+1}.pt"
            ckpt_path = os.path.join(out_dir, ckpt_name)
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': {
                    'vocab_size': vocab_size, 'block_size': block_size,
                    'n_layers': n_layers, 'n_heads': n_heads, 'n_embd': n_embd,
                    'with_layer_norm': with_ln, 'without_pos': True,
                    'learning_rate': learning_rate, 'max_iters': max_iters,
                    'batch_size': batch_size, 'warmup_iters': warmup_iters,
                    'min_lr': min_lr, 'weight_decay': weight_decay,
                    'data_seed': data_seed, 'init_seed': init_seed,
                },
                'itr': itr + 1,
                'train_loss': loss.item(),
            }, ckpt_path)
            print(f"  Saved {ckpt_path}", flush=True)

    elapsed = time.time() - t0
    print(f"Finished: {run_tag} in {elapsed:.0f}s")


if __name__ == '__main__':
    main()
