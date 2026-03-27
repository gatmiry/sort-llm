"""
Single training run for grid search. Accepts all hyperparameters via CLI.
Fixed params: n_layers=2, n_heads=1, n_embd=64, batch_size=4096, without_pos=True,
              warmup_iters=100, min_lr=1e-6, weight_decay=0.1
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
    p.add_argument('--with_layer_norm', type=int, required=True, choices=[0, 1])
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--output_dir', type=str, required=True)
    p.add_argument('--seed', type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    device = 'cuda:0'
    torch.cuda.set_device(0)

    vocab_size = args.vocab_size
    block_size = args.block_size
    max_iters = args.max_iters
    learning_rate = args.learning_rate
    with_ln = bool(args.with_layer_norm)

    # Fixed params
    batch_size = 4096
    n_layers = 2
    n_heads = 1
    n_embd = 64
    warmup_iters = 100
    min_lr = 1e-6
    weight_decay = 0.1
    total_iters = max_iters
    ckpt_interval = 2000

    # For large vocab, reduce micro-batch and use gradient accumulation
    micro_batch = batch_size
    accum_steps = 1
    if vocab_size >= 8192:
        micro_batch = 256
        accum_steps = batch_size // micro_batch
    elif vocab_size >= 512:
        micro_batch = 1024
        accum_steps = batch_size // micro_batch

    os.makedirs(args.output_dir, exist_ok=True)

    def get_batch(bs):
        def cat_sorted(x):
            vals, _ = torch.sort(x)
            return torch.cat((x, torch.tensor([vocab_size]), vals), dim=0)
        return torch.stack([cat_sorted(torch.randperm(vocab_size)[:block_size]) for _ in range(bs)])

    def get_lr(itr):
        if itr < warmup_iters:
            return learning_rate * (itr + 1) / (warmup_iters + 1)
        if itr > max_iters:
            return min_lr
        ratio = (itr - warmup_iters) / (max_iters - warmup_iters)
        ratio = 0.5 * (1.0 + math.cos(math.pi * ratio))
        return min_lr + ratio * (learning_rate - min_lr)

    config = GPTConfig(block_size=block_size, vocab_size=vocab_size, with_layer_norm=with_ln)
    model = GPT(config)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    decay_params = [p for p in params if p.dim() > 1]
    nondecay_params = [p for p in params if p.dim() <= 1]
    optimizer = torch.optim.AdamW([
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nondecay_params, 'weight_decay': 0.0}
    ], lr=learning_rate, betas=(0.9, 0.95), eps=1e-8)

    ln_str = "with" if with_ln else "without"
    lr_str = f"{learning_rate:.0e}"
    run_tag = f"V{vocab_size}_B{block_size}_LR{lr_str}_MI{max_iters}_LN{args.with_layer_norm}_E{n_embd}_H{n_heads}_L{n_layers}"
    print(f"[GPU {args.gpu}] Starting: {run_tag}")
    t0 = time.time()

    for itr in range(total_iters):
        model.train()
        optimizer.zero_grad()

        for astep in range(accum_steps):
            x = get_batch(micro_batch).to(device)
            logits, loss = model(x)
            (loss / accum_steps).backward()

        lr = get_lr(itr)
        for pg in optimizer.param_groups:
            pg['lr'] = lr
        optimizer.step()

        if itr % 500 == 0:
            model.eval()
            with torch.no_grad():
                x_test = get_batch(min(micro_batch, 512)).to(device)
                _, test_loss = model(x_test)
            elapsed = time.time() - t0
            print(f"[GPU {args.gpu}] {run_tag} | itr {itr}/{total_iters} | train_loss {loss.item():.4f} | test_loss {test_loss.item():.4f} | lr {lr:.2e} | {elapsed:.0f}s")

        if (itr + 1) % ckpt_interval == 0 or itr == total_iters - 1:
            ckpt_name = f"{run_tag}_itr{itr+1}.pt"
            ckpt_path = os.path.join(args.output_dir, ckpt_name)
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
                },
                'itr': itr + 1,
                'train_loss': loss.item(),
            }, ckpt_path)
            print(f"[GPU {args.gpu}] Saved {ckpt_path}")

    elapsed = time.time() - t0
    print(f"[GPU {args.gpu}] Finished: {run_tag} in {elapsed:.0f}s")


if __name__ == '__main__':
    main()
