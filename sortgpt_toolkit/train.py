#!/usr/bin/env python3
"""
Train a single SortGPT model.

Usage:
    CUDA_VISIBLE_DEVICES=X python train.py \\
        --init-seed 1501 --init-std 0.02 --run-dir ./my_run \\
        --max-iters 100000 --checkpoint-every 20000

All model/training hyperparameters can be set via command-line flags.
"""

import argparse
import gc
import math
import os
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import trange

from model import (
    DEVICE, AMP_DTYPE, GPT, GPTConfig,
    make_generator, autocast_ctx, make_grad_scaler,
    get_batch, get_lr, create_optimizer, float_token,
)


def main():
    parser = argparse.ArgumentParser(description="Train a single SortGPT model")

    # Required
    parser.add_argument("--init-seed", type=int, required=True,
                        help="Seed for weight initialization + RNG")
    parser.add_argument("--init-std", type=float, required=True,
                        help="Standard deviation for weight initialization")
    parser.add_argument("--run-dir", type=str, required=True,
                        help="Output directory (checkpoints saved to run_dir/checkpoints/)")

    # Training duration
    parser.add_argument("--max-iters", type=int, default=100_000,
                        help="Total training iterations (default: 100k)")
    parser.add_argument("--checkpoint-every", type=int, default=20_000,
                        help="Save checkpoint every N iterations (default: 20k)")

    # Hyperparameters
    parser.add_argument("--lr", type=float, default=0.03, help="Peak learning rate")
    parser.add_argument("--batch-size", type=int, default=4096, help="Effective batch size")
    parser.add_argument("--warmup-iters", type=int, default=200, help="LR warmup iterations")
    parser.add_argument("--min-lr", type=float, default=1e-6, help="Minimum learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay")

    # Architecture
    parser.add_argument("--block-size", type=int, default=16, help="Sequence length k")
    parser.add_argument("--vocab-n", type=int, default=256, help="Vocabulary size")
    parser.add_argument("--n-layers", type=int, default=2, help="Number of transformer layers")
    parser.add_argument("--n-heads", type=int, default=1, help="Number of attention heads")
    parser.add_argument("--n-embd", type=int, default=64, help="Embedding dimension")
    parser.add_argument("--use-pos", action="store_true", help="Enable positional embeddings")
    parser.add_argument("--no-mlp", action="store_true", help="Disable MLP layers")
    parser.add_argument("--no-final-ln", action="store_true", help="Disable final LayerNorm")

    # Per-layer init scaling
    parser.add_argument("--l1-init-scale", type=float, default=1.0,
                        help="Multiply init std of second transformer block (h[1]) by this factor")

    # Data
    parser.add_argument("--data-seed", type=int, default=1337, help="Seed for data generation")
    parser.add_argument("--allow-duplicates", action="store_true", help="Allow duplicate tokens")

    # Logging
    parser.add_argument("--log-interval", type=int, default=1000, help="Log every N iterations")

    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    init_seed = args.init_seed
    init_std = args.init_std
    model_tag = f"std{float_token(init_std)}_iseed{init_seed}"

    gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    print(f"=== Training {model_tag} on GPU {gpu_id} for {args.max_iters} iters ===")
    print(f"    DEVICE={DEVICE}, AMP_DTYPE={AMP_DTYPE}")
    print(f"    init_std={init_std}, lr={args.lr}, batch_size={args.batch_size}")
    print(f"    k={args.block_size}, N={args.vocab_n}, E={args.n_embd}, L={args.n_layers}")

    # Set seeds
    torch.manual_seed(init_seed)
    np.random.seed(init_seed)
    random.seed(init_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(init_seed)

    train_generator = make_generator(DEVICE, args.data_seed)

    max_supported_length = args.block_size * 6
    max_seq_len = 2 * max_supported_length + 1

    model_cfg = GPTConfig(
        block_size=args.block_size,
        vocab_size=args.vocab_n + 1,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_embd=args.n_embd,
        without_pos=not args.use_pos,
        use_mlp=not args.no_mlp,
        use_final_LN=not args.no_final_ln,
        max_seq_len=max_seq_len,
    )

    GPT._init_std = init_std
    model = GPT(model_cfg).to(DEVICE)

    if args.l1_init_scale != 1.0 and model_cfg.n_layers >= 2:
        scaled_std = init_std * args.l1_init_scale
        with torch.no_grad():
            for name, param in model.transformer.h[1].named_parameters():
                if "ln_" not in name:
                    nn.init.normal_(param, mean=0, std=scaled_std)
                    if "bias" in name:
                        nn.init.zeros_(param)
        print(f"    Rescaled h[1] weights: std={init_std} * {args.l1_init_scale} = {scaled_std}")

    scaler = make_grad_scaler(enabled=(DEVICE.type == "cuda" and AMP_DTYPE == torch.float16))
    optimizer = create_optimizer(model, weight_decay=args.weight_decay, lr=args.lr)

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    for itr in trange(args.max_iters, desc=f"train {model_tag}", leave=True):
        lr = get_lr(itr, args.max_iters, args.lr, args.warmup_iters, args.min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)

        idx = get_batch(
            batch_size=args.batch_size,
            length=args.block_size,
            device=DEVICE,
            vocab_n=args.vocab_n,
            allow_duplicates=args.allow_duplicates,
            generator=train_generator,
        )
        with autocast_ctx(DEVICE, enabled=True):
            logits, loss = model(idx, block_size=args.block_size, return_full_logits=False)

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        do_log = ((itr + 1) % args.log_interval == 0) or (itr == 0) or (itr + 1 == args.max_iters)
        if do_log:
            with torch.no_grad():
                targets = idx[:, args.block_size + 1:]
                preds = logits.detach().argmax(dim=-1)
                tok_acc = float((preds == targets).sum().item()) / float(targets.numel())
                samp_acc = float((preds == targets).all(dim=1).sum().item()) / float(targets.size(0))
            print(f"  itr={itr+1:7d} | lr={lr:.3e} | loss={loss.item():.6f} "
                  f"| tok_acc={tok_acc:.4f} | samp_acc={samp_acc:.4f}")

        if (itr + 1) % args.checkpoint_every == 0 or (itr + 1) == args.max_iters:
            ckpt_name = f"{model_tag}__ckpt{itr+1}.pt"
            ckpt_path = ckpt_dir / ckpt_name
            cpu_sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            torch.save({
                "artifact_type": "checkpoint",
                "checkpoint_iter": int(itr + 1),
                "init_seed": init_seed,
                "init_std": init_std,
                "l1_init_scale": args.l1_init_scale,
                "model_config": asdict(model_cfg),
                "model_state_dict": cpu_sd,
            }, ckpt_path)
            del cpu_sd
            elapsed = (time.perf_counter() - t0) / 60.0
            print(f"  [checkpoint saved] {ckpt_name}  ({elapsed:.1f} min elapsed)")

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    total_min = (time.perf_counter() - t0) / 60.0
    print(f"\n=== Done: {model_tag} in {total_min:.1f} minutes ===")


if __name__ == "__main__":
    main()
