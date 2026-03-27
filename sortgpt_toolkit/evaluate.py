"""
Evaluation utilities: ablation analysis and length generalization.
"""

import math
from contextlib import contextmanager

import torch
import torch.nn as nn

from model import (
    DEVICE, AMP_DTYPE, autocast_ctx, make_generator, get_batch
)

# ── Defaults ──────────────────────────────────────────────────────────────────

EVAL_BATCH_SIZE = 128
EVAL_BATCHES = 20
EVAL_LENGTHS = [4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 72, 80, 88, 96,
                112, 128, 144, 160, 176, 192, 208, 224, 240, 256]


# ── Ablation ──────────────────────────────────────────────────────────────────

class _ZeroAttnOutput(nn.Module):
    def forward(self, x):
        return torch.zeros_like(x)


@contextmanager
def ablate_attention(model, layer_idx):
    """Temporarily replace attention in a given layer with zeros."""
    block = model.transformer.h[layer_idx]
    original_attn = block.attn
    block.attn = _ZeroAttnOutput().to(DEVICE)
    try:
        yield
    finally:
        block.attn = original_attn


@torch.no_grad()
def evaluate_token_accuracy(model, block_size, vocab_n, *,
                            n_batches=EVAL_BATCHES, batch_size=EVAL_BATCH_SIZE,
                            generator_seed=42_000):
    """
    Evaluate token-level and sample-level accuracy on the sorting task.

    Returns: (token_accuracy, sample_accuracy)
    """
    model.eval()
    token_correct = token_total = sample_correct = sample_total = 0
    gen = make_generator(DEVICE, generator_seed)
    for _ in range(n_batches):
        idx = get_batch(batch_size, block_size, DEVICE,
                        vocab_n=vocab_n, allow_duplicates=False, generator=gen)
        with autocast_ctx(DEVICE, enabled=True):
            logits, _ = model(idx, block_size=block_size, return_full_logits=False)
        targets = idx[:, block_size + 1:]
        preds = logits.detach().argmax(dim=-1)
        token_correct += int((preds == targets).sum().item())
        token_total += int(targets.numel())
        sample_correct += int((preds == targets).all(dim=1).sum().item())
        sample_total += int(targets.size(0))
    tok_acc = float(token_correct / max(token_total, 1))
    samp_acc = float(sample_correct / max(sample_total, 1))
    return tok_acc, samp_acc


def evaluate_ablation(model, block_size, vocab_n, **kwargs):
    """
    Evaluate full model + ablation of each attention layer.

    Returns dict with keys:
        full_tok, full_samp, no_attn1_tok, no_attn1_samp, no_attn2_tok, no_attn2_samp
    """
    full_tok, full_samp = evaluate_token_accuracy(model, block_size, vocab_n, **kwargs)
    with ablate_attention(model, 0):
        no1_tok, no1_samp = evaluate_token_accuracy(model, block_size, vocab_n, **kwargs)
    with ablate_attention(model, 1):
        no2_tok, no2_samp = evaluate_token_accuracy(model, block_size, vocab_n, **kwargs)
    return {
        "full_tok": full_tok, "full_samp": full_samp,
        "no_attn1_tok": no1_tok, "no_attn1_samp": no1_samp,
        "no_attn2_tok": no2_tok, "no_attn2_samp": no2_samp,
    }


# ── Length generalization ────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_at_length(model, length, vocab_n, *,
                       n_batches=EVAL_BATCHES, batch_size=EVAL_BATCH_SIZE,
                       generator_seed=42_000):
    """Evaluate token accuracy at a specific sequence length."""
    model.eval()
    token_correct = token_total = 0
    gen = make_generator(DEVICE, int(generator_seed + 97 * length))
    for _ in range(n_batches):
        idx = get_batch(batch_size, length, DEVICE,
                        vocab_n=vocab_n, allow_duplicates=False, generator=gen)
        with autocast_ctx(DEVICE, enabled=True):
            logits, _ = model(idx, block_size=length, return_full_logits=False)
        targets = idx[:, length + 1:]
        preds = logits.detach().argmax(dim=-1)
        token_correct += int((preds == targets).sum().item())
        token_total += int(targets.numel())
    return float(token_correct / max(token_total, 1))


def evaluate_length_generalization(model, vocab_n, lengths=None, **kwargs):
    """
    Evaluate token accuracy across multiple sequence lengths.

    Returns list of accuracies, one per length.
    """
    if lengths is None:
        lengths = EVAL_LENGTHS
    return [evaluate_at_length(model, L, vocab_n, **kwargs) for L in lengths]
