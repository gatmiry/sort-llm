"""
Attention intervention analysis for SortGPT models.

Provides tools to:
  - Expose raw (pre-softmax) attention weights from the toolkit's model
  - Intervene on attention patterns (intensity sweep, hijack, aggressive SEP)
  - Run ablation and baseline accuracy analyses
  - Compute logit-vs-attention agreement (cinclogits)

Usage as CLI:
    python intervene.py --ckpt PATH --task intensity --layer 0 --out result.npz
    python intervene.py --ckpt PATH --task hijack   --layer 0 --out result.npz
    python intervene.py --ckpt PATH --task ablation  --layer 0 --out result.npz
    python intervene.py --ckpt PATH --task baseline  --out result.npz
    python intervene.py --ckpt PATH --task cinclogits --layer 0 --out result.npz
    python intervene.py --ckpt PATH --task aggressive_sep --layer 0 --out result.npz

Usage as library:
    from intervene import GPTIntervention, enable_attention_storage
    from model import load_model_from_checkpoint, DEVICE

    model = load_model_from_checkpoint("checkpoint.pt")
    block_size = model.config.block_size
    vocab_n = model.config.vocab_size - 1

    enable_attention_storage(model)
    idx = get_single_batch(vocab_n, block_size, DEVICE)
    im = GPTIntervention(model, idx, block_size=block_size)
    im.intervent_attention(attention_layer_num=0, location=block_size+5, ...)
    pred, target = im.check_if_still_works()
    im.revert_attention(0)
    disable_attention_storage(model)
"""

import argparse
import math
import os
import types

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import (
    DEVICE, autocast_ctx, get_batch as _toolkit_get_batch,
    load_model_from_checkpoint, GPT,
)


# ── Attention storage ────────────────────────────────────────────────────────
# The toolkit's CausalSelfAttention uses F.scaled_dot_product_attention, which
# is fused and does not expose attention weights.  These helpers replace the
# forward with a manual implementation that stores raw (pre-softmax) and
# post-softmax attention matrices, which the intervention code needs.

def _manual_attn_forward(self, x):
    """Drop-in replacement for CausalSelfAttention.forward that stores
    raw_attn (pre-softmax) and attn (post-softmax) as (T, T) tensors."""
    B, T, C = x.size()
    qkv = self.c_attn(x)
    q, k, v = qkv.split(self.n_embd, dim=2)
    q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
    k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
    v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

    scale = 1.0 / math.sqrt(self.head_dim)
    att = (q @ k.transpose(-2, -1)) * scale

    causal_mask = torch.triu(
        torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
    )
    att = att.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
    self.raw_attn = att.clone().detach().squeeze(0).squeeze(0)

    att = F.softmax(att, dim=-1)
    self.attn = att.clone().detach().squeeze(0).squeeze(0)

    y = att @ v
    y = y.transpose(1, 2).contiguous().view(B, T, C)
    return self.c_proj(y)


def enable_attention_storage(model):
    """Patch all attention layers to store raw/post-softmax attention weights.

    Call this before running any intervention or cinclogits analysis.
    """
    for block in model.transformer.h:
        attn_mod = block.attn
        if not hasattr(attn_mod, "_orig_forward"):
            attn_mod._orig_forward = attn_mod.forward
            attn_mod.forward = types.MethodType(_manual_attn_forward, attn_mod)


def disable_attention_storage(model):
    """Restore the original fused attention forward on all layers."""
    for block in model.transformer.h:
        attn_mod = block.attn
        if hasattr(attn_mod, "_orig_forward"):
            attn_mod.forward = attn_mod._orig_forward
            del attn_mod._orig_forward


# ── Data helpers ─────────────────────────────────────────────────────────────
# The toolkit's vocab_size includes the SEP token (vocab_size = vocab_n + 1).
# These helpers provide the same simple batch generation used in the
# intervention scripts (single sample, no duplicates).

def get_single_batch(vocab_n, block_size, device):
    """Generate one sample: [unsorted | SEP | sorted], shape (1, 2*block_size+1)."""
    x = torch.randperm(vocab_n)[:block_size]
    vals, _ = torch.sort(x)
    sep = torch.tensor([vocab_n])
    return torch.cat((x, sep, vals), dim=0).unsqueeze(0).to(device)


# ── GPTIntervention ──────────────────────────────────────────────────────────

class GPTIntervention:
    """Monkey-patch a single attention layer to inject pre-softmax logit edits.

    Mirrors the interface of the original GPTIntervention from model_analysis.py
    but works with the sortgpt_toolkit model architecture.

    Typical usage::

        enable_attention_storage(model)
        idx = get_single_batch(vocab_n, block_size, device)
        im = GPTIntervention(model, idx, block_size=block_size)
        im.intervent_attention(
            attention_layer_num=0, location=block_size + 5,
            unsorted_lb=5, unsorted_ub=5,
            unsorted_lb_num=0, unsorted_ub_num=1,
            unsorted_intensity_inc=1.0,
            sorted_lb=0, sorted_num=0, sorted_intensity_inc=0.0,
        )
        pred, target = im.check_if_still_works()
        im.revert_attention(0)
    """

    def __init__(self, model, idx, *, block_size):
        self.model = model
        self.idx = idx
        self.block_size = block_size
        self.n_layers = model.config.n_layers

        with torch.no_grad():
            self.model(self.idx, block_size=self.block_size)

        self.attn = [
            model.transformer.h[i].attn.attn for i in range(self.n_layers)
        ]
        self.raw_attn = [
            model.transformer.h[i].attn.raw_attn for i in range(self.n_layers)
        ]
        self._old_forwards = [None] * self.n_layers

    def read_attention(self, layer, loc1, loc2):
        return self.raw_attn[layer][loc1, loc2]

    def check_if_still_works(self):
        """Run a forward pass and return (predicted_token, target_token) at self.location."""
        with torch.no_grad():
            logits, _ = self.model(self.idx, block_size=self.block_size,
                                   return_full_logits=True)
        predicted = torch.argmax(logits, dim=-1)[0, self.location].item()
        target = self.idx[0, self.location + 1].item()
        return predicted, target

    def intervent_attention(self, attention_layer_num, location,
                            unsorted_lb, unsorted_ub,
                            unsorted_lb_num, unsorted_ub_num,
                            unsorted_intensity_inc,
                            sorted_lb, sorted_num, sorted_intensity_inc):
        """Inject pre-softmax attention edits for wrong keys near the correct one.

        At query position *location* in the sorted half, find wrong keys in
        the unsorted prefix that are within value windows around the target,
        set their pre-softmax attention logit to ``main_attention_val + intensity``,
        where *main_attention_val* is the logit of the correct next key.
        """
        self.location = location
        bs = self.block_size
        target_val = self.idx[0, location].item()
        next_number = self.idx[0, location + 1].item()
        unsorted_part = self.idx[0, :bs]
        sorted_part = self.idx[0, bs + 1: 2 * bs + 1]

        # --- select wrong keys below target ---
        lb_mask = (
            (unsorted_part >= target_val - unsorted_lb)
            & (unsorted_part <= target_val)
            & (unsorted_part != next_number)
        )
        lb_indices = torch.where(lb_mask)[0]
        if len(lb_indices) < unsorted_lb_num:
            raise ValueError(
                f"Not enough numbers for unsorted_lb_num "
                f"(need {unsorted_lb_num}, got {len(lb_indices)})"
            )
        lb_selected = lb_indices[torch.randperm(len(lb_indices))[:unsorted_lb_num]]

        # --- select wrong keys above target ---
        ub_mask = (
            (unsorted_part > target_val)
            & (unsorted_part <= target_val + unsorted_ub)
            & (unsorted_part != next_number)
        )
        ub_indices = torch.where(ub_mask)[0]
        if len(ub_indices) < unsorted_ub_num:
            raise ValueError(
                f"Not enough numbers for unsorted_ub_num "
                f"(need {unsorted_ub_num}, got {len(ub_indices)})"
            )
        ub_selected = (
            ub_indices[torch.randperm(len(ub_indices))[:unsorted_ub_num]]
            if len(ub_indices) > 0 else torch.tensor([], dtype=torch.long)
        )

        # --- select wrong keys in sorted region ---
        sorted_mask = torch.abs(sorted_part - target_val) <= sorted_lb
        sorted_indices = torch.where(sorted_mask)[0]
        if len(sorted_indices) < sorted_num:
            raise ValueError(
                f"Not enough numbers for sorted_num "
                f"(need {sorted_num}, got {len(sorted_indices)})"
            )
        sorted_selected = sorted_indices[torch.randperm(len(sorted_indices))[:sorted_num]]
        sorted_actual_indices = sorted_selected + bs + 1

        # baseline logit from the correct key
        next_loc = torch.where(self.idx[0, :bs] == next_number)[0][0].item()
        main_attn_val = self.read_attention(attention_layer_num, location, next_loc).item()

        # capture for closure
        attn_mod = self.model.transformer.h[attention_layer_num].attn

        def new_forward(self_attn, x):
            B, T, C = x.size()
            qkv = self_attn.c_attn(x)
            q, k, v = qkv.split(self_attn.n_embd, dim=2)
            q = q.view(B, T, self_attn.n_heads, self_attn.head_dim).transpose(1, 2)
            k = k.view(B, T, self_attn.n_heads, self_attn.head_dim).transpose(1, 2)
            v = v.view(B, T, self_attn.n_heads, self_attn.head_dim).transpose(1, 2)

            scale = 1.0 / math.sqrt(self_attn.head_dim)
            att = (q @ k.transpose(-2, -1)) * scale

            for idx_t in lb_selected:
                att[:, :, location, idx_t.item()] = main_attn_val + unsorted_intensity_inc
            for idx_t in ub_selected:
                att[:, :, location, idx_t.item()] = main_attn_val + unsorted_intensity_inc
            for idx_t in sorted_actual_indices:
                att[:, :, location, idx_t.item()] = main_attn_val + sorted_intensity_inc

            causal = torch.triu(
                torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
            )
            att = att.masked_fill(causal.unsqueeze(0).unsqueeze(0), float("-inf"))
            att = F.softmax(att, dim=-1)
            y = att @ v
            y = y.transpose(1, 2).contiguous().view(B, T, C)
            return self_attn.c_proj(y)

        self._old_forwards[attention_layer_num] = attn_mod.forward
        attn_mod.forward = types.MethodType(new_forward, attn_mod)

    def revert_attention(self, attention_layer_num):
        if self._old_forwards[attention_layer_num] is None:
            raise RuntimeError("No saved forward to revert to")
        attn_mod = self.model.transformer.h[attention_layer_num].attn
        attn_mod.forward = self._old_forwards[attention_layer_num]
        self._old_forwards[attention_layer_num] = None


# ── Analysis tasks ───────────────────────────────────────────────────────────

def compute_baseline(model, block_size, vocab_n, device, num_trials=500):
    """Intact model accuracy with teacher forcing."""
    per_pos_correct = np.zeros(block_size)
    full_seq_correct = 0
    cond_correct = np.zeros(block_size)
    cond_eligible = np.zeros(block_size)

    for _ in range(num_trials):
        idx = get_single_batch(vocab_n, block_size, device)
        with torch.no_grad():
            logits, _ = model(idx, block_size=block_size, return_full_logits=True)
        preds = torch.argmax(logits[0, block_size:2 * block_size, :], dim=1)
        targets = idx[0, block_size + 1:]
        correct = (preds == targets).cpu().numpy()
        per_pos_correct += correct
        if correct.all():
            full_seq_correct += 1
        prefix_ok = True
        for i in range(block_size):
            if prefix_ok:
                cond_eligible[i] += 1
                if correct[i]:
                    cond_correct[i] += 1
                else:
                    prefix_ok = False
            else:
                break

    return {
        "per_pos_acc": per_pos_correct / num_trials,
        "full_seq_acc": full_seq_correct / num_trials,
        "cond_acc": np.divide(cond_correct, cond_eligible,
                             out=np.zeros_like(cond_correct),
                             where=cond_eligible > 0),
        "cond_eligible": cond_eligible,
    }


def compute_ablation(model, block_size, vocab_n, device, skip_layer, num_trials=500):
    """Accuracy when bypassing attention+layernorm for a layer."""
    block = model.transformer.h[skip_layer]
    original_forward = block.forward

    def forward_skip_attn(x):
        if block.use_mlp and block.mlp is not None:
            return x + block.mlp(block.ln_2(x))
        return x

    block.forward = forward_skip_attn

    try:
        result = compute_baseline(model, block_size, vocab_n, device, num_trials)
    finally:
        block.forward = original_forward

    result["skip_layer"] = skip_layer
    return result


def compute_cinclogits(model, block_size, vocab_n, device, attn_layer, num_tries=100):
    """Measure agreement between argmax-logit and argmax-attention at each position."""
    acc_clogit_icscore = np.zeros(block_size)
    acc_iclogit_icscore = np.zeros(block_size)

    for _ in range(num_tries):
        idx = get_single_batch(vocab_n, block_size, device)
        with torch.no_grad():
            logits, _ = model(idx, block_size=block_size, return_full_logits=True)

        is_correct = (
            torch.argmax(logits[0, block_size:2 * block_size, :], dim=1)
            == idx[0, block_size + 1:]
        )
        attn_weights = model.transformer.h[attn_layer].attn.attn
        seq_len = 2 * block_size + 1

        for j in range(block_size, 2 * block_size):
            max_score = float("-inf")
            max_score_num = -1
            for k_pos in range(seq_len):
                score = attn_weights[j, k_pos].item()
                if score > max_score:
                    max_score = score
                    max_score_num = idx[0, k_pos].item()
            score_correct = max_score_num == idx[0, j + 1].item()
            pos = j - block_size
            logit_correct = is_correct[pos].item()
            if logit_correct and not score_correct:
                acc_clogit_icscore[pos] += 1.0
            elif not logit_correct and not score_correct:
                acc_iclogit_icscore[pos] += 1.0

    return acc_clogit_icscore / num_tries, acc_iclogit_icscore / num_tries


def compute_intensity(model, block_size, vocab_n, device, attn_layer,
                      unsorted_lb=5, unsorted_ub=5, min_valid=200,
                      intensity_values=None):
    """Sweep intervention intensity and measure how often the model still predicts correctly."""
    if intensity_values is None:
        intensity_values = [
            -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0,
        ]
    location = block_size + 5

    success_rates = []
    counts = []
    for intensity in intensity_values:
        attempts = []
        rounds = 0
        while len(attempts) < min_valid and rounds < 2000:
            rounds += 1
            idx = get_single_batch(vocab_n, block_size, device)
            try:
                im = GPTIntervention(model, idx, block_size=block_size)
                im.intervent_attention(
                    attention_layer_num=attn_layer, location=location,
                    unsorted_lb=unsorted_lb, unsorted_ub=unsorted_ub,
                    unsorted_lb_num=0, unsorted_ub_num=1,
                    unsorted_intensity_inc=intensity,
                    sorted_lb=0, sorted_num=0, sorted_intensity_inc=0.0,
                )
                pred, target = im.check_if_still_works()
                attempts.append(pred == target)
                im.revert_attention(attn_layer)
            except (ValueError, IndexError):
                continue
        n = len(attempts)
        counts.append(n)
        if n < min_valid:
            print(f"  WARNING: intensity={intensity:.2f} got {n}/{min_valid} valid "
                  f"after 2000 rounds", flush=True)
        success_rates.append(sum(attempts) / n if n > 0 else 0.0)

    return np.array(intensity_values), np.array(success_rates), np.array(counts)


def compute_hijack(model, block_size, vocab_n, device, attn_layer,
                   n_trials=2000, hijack_intensity=10.0):
    """Boost attention to a random wrong key and record what the model predicts.

    Returns ndarray of shape (N, 4) with columns:
        [current_sorted_value, boosted_number, predicted, correct_next]
    """
    attn_mod = model.transformer.h[attn_layer].attn
    records = []

    for _ in range(n_trials):
        idx = get_single_batch(vocab_n, block_size, device)
        unsorted = idx[0, :block_size]
        sorted_part = idx[0, block_size + 1: 2 * block_size + 1]

        with torch.no_grad():
            model(idx, block_size=block_size)
        raw_attn = attn_mod.raw_attn.clone()

        for p in range(block_size - 1):
            location = block_size + 1 + p
            current_num = sorted_part[p].item()
            correct_next = idx[0, location + 1].item()

            next_in_unsorted = (unsorted == correct_next).nonzero(as_tuple=True)[0]
            if len(next_in_unsorted) == 0:
                continue
            next_loc = next_in_unsorted[0].item()
            main_attn_val = raw_attn[location, next_loc].item()

            candidates = [i for i in range(block_size) if unsorted[i].item() != correct_next]
            if not candidates:
                continue
            boost_idx = candidates[torch.randint(len(candidates), (1,)).item()]
            boosted_number = unsorted[boost_idx].item()

            def _make_fwd(loc, bidx, mav, intens):
                def fwd(self_attn, x):
                    B, T, C = x.size()
                    qkv = self_attn.c_attn(x)
                    q, k, v = qkv.split(self_attn.n_embd, dim=2)
                    q = q.view(B, T, self_attn.n_heads, self_attn.head_dim).transpose(1, 2)
                    k = k.view(B, T, self_attn.n_heads, self_attn.head_dim).transpose(1, 2)
                    v = v.view(B, T, self_attn.n_heads, self_attn.head_dim).transpose(1, 2)
                    scale = 1.0 / math.sqrt(self_attn.head_dim)
                    att = (q @ k.transpose(-2, -1)) * scale
                    att[:, :, loc, bidx] = mav + intens
                    causal = torch.triu(
                        torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
                    )
                    att = att.masked_fill(causal.unsqueeze(0).unsqueeze(0), float("-inf"))
                    att = F.softmax(att, dim=-1)
                    y = att @ v
                    y = y.transpose(1, 2).contiguous().view(B, T, C)
                    return self_attn.c_proj(y)
                return fwd

            old_fwd = attn_mod.forward
            attn_mod.forward = types.MethodType(
                _make_fwd(location, boost_idx, main_attn_val, hijack_intensity), attn_mod
            )
            with torch.no_grad():
                logits, _ = model(idx, block_size=block_size, return_full_logits=True)
            predicted = torch.argmax(logits, dim=-1)[0, location].item()
            attn_mod.forward = old_fwd
            records.append((current_num, boosted_number, predicted, correct_next))

    return np.array(records, dtype=np.int32) if records else np.empty((0, 4), dtype=np.int32)


def compute_aggressive_sep(model, block_size, vocab_n, device, attn_layer,
                           num_trials=500, intensity_values=None):
    """For every sorting position, set attention to SEP = attention(correct) + intensity.

    Returns dict mapping intensity -> token_success_rate.
    """
    if intensity_values is None:
        intensity_values = [
            -4.0, -3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
        ]
    sep_pos = block_size
    attn_mod = model.transformer.h[attn_layer].attn
    original_fwd = attn_mod.forward
    current_idx = [None]

    results = {}
    for intensity in intensity_values:
        correct_tokens = 0
        total_tokens = 0

        def _make_fwd(intens):
            def fwd(self_attn, x):
                B, T, C = x.size()
                qkv = self_attn.c_attn(x)
                q, k, v = qkv.split(self_attn.n_embd, dim=2)
                q = q.view(B, T, self_attn.n_heads, self_attn.head_dim).transpose(1, 2)
                k = k.view(B, T, self_attn.n_heads, self_attn.head_dim).transpose(1, 2)
                v = v.view(B, T, self_attn.n_heads, self_attn.head_dim).transpose(1, 2)
                scale = 1.0 / math.sqrt(self_attn.head_dim)
                att = (q @ k.transpose(-2, -1)) * scale

                idx_t = current_idx[0]
                bs = block_size
                for pos in range(bs, 2 * bs):
                    next_num = idx_t[0, pos + 1].item()
                    next_loc = (idx_t[0, :bs] == next_num).nonzero(as_tuple=True)[0][0].item()
                    main_val = att[:, :, pos, next_loc].clone()
                    att[:, :, pos, sep_pos] = main_val + intens

                causal = torch.triu(
                    torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
                )
                att = att.masked_fill(causal.unsqueeze(0).unsqueeze(0), float("-inf"))
                att = F.softmax(att, dim=-1)
                y = att @ v
                y = y.transpose(1, 2).contiguous().view(B, T, C)
                return self_attn.c_proj(y)
            return fwd

        attn_mod.forward = types.MethodType(_make_fwd(intensity), attn_mod)
        for _ in range(num_trials):
            idx = get_single_batch(vocab_n, block_size, device)
            current_idx[0] = idx
            with torch.no_grad():
                logits, _ = model(idx, block_size=block_size, return_full_logits=True)
            preds = torch.argmax(logits[0, block_size:2 * block_size, :], dim=1)
            targets = idx[0, block_size + 1:]
            correct_tokens += (preds == targets).sum().item()
            total_tokens += len(targets)
        attn_mod.forward = original_fwd
        rate = correct_tokens / total_tokens
        results[intensity] = rate
        print(f"    intensity={intensity:+.2f}: {rate:.4f} "
              f"({correct_tokens}/{total_tokens})", flush=True)
    return results


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run intervention analysis on a sortgpt_toolkit checkpoint"
    )
    parser.add_argument("--ckpt", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--task", required=True,
                        choices=["intensity", "hijack", "ablation",
                                 "baseline", "cinclogits", "aggressive_sep"])
    parser.add_argument("--layer", type=int, default=0,
                        help="Attention layer to intervene on (default: 0)")
    parser.add_argument("--out", required=True, help="Output .npz path")
    parser.add_argument("--device", default=None,
                        help="Device (default: auto-detect)")
    parser.add_argument("--unsorted-lb", type=int, default=5)
    parser.add_argument("--unsorted-ub", type=int, default=5)
    parser.add_argument("--num-trials", type=int, default=500)
    args = parser.parse_args()

    device = args.device or str(DEVICE)
    if device != str(DEVICE):
        import model as _m
        _m.DEVICE = torch.device(device)

    model = load_model_from_checkpoint(args.ckpt)
    block_size = model.config.block_size
    vocab_n = model.config.vocab_size - 1
    enable_attention_storage(model)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    if args.task == "baseline":
        r = compute_baseline(model, block_size, vocab_n, DEVICE, args.num_trials)
        np.savez(args.out, **r)
        print(f"  baseline: full_seq_acc={r['full_seq_acc']:.4f}", flush=True)

    elif args.task == "ablation":
        r = compute_ablation(model, block_size, vocab_n, DEVICE,
                             skip_layer=args.layer, num_trials=args.num_trials)
        np.savez(args.out, **{k: v for k, v in r.items()})
        print(f"  ablation layer={args.layer}: "
              f"full_seq_acc={r['full_seq_acc']:.4f}", flush=True)

    elif args.task == "cinclogits":
        cl_ic, icl_ic = compute_cinclogits(
            model, block_size, vocab_n, DEVICE, args.layer, num_tries=args.num_trials
        )
        np.savez(args.out, clogit_icscore=cl_ic, iclogit_icscore=icl_ic)
        print(f"  cinclogits layer={args.layer}: done", flush=True)

    elif args.task == "intensity":
        intensities, rates, counts = compute_intensity(
            model, block_size, vocab_n, DEVICE, args.layer,
            unsorted_lb=args.unsorted_lb, unsorted_ub=args.unsorted_ub,
        )
        np.savez(args.out, intensities=intensities,
                 success_rates=rates, counts=counts)
        print(f"  intensity layer={args.layer}: done", flush=True)

    elif args.task == "hijack":
        records = compute_hijack(
            model, block_size, vocab_n, DEVICE, args.layer,
            n_trials=args.num_trials,
        )
        np.savez(args.out, data=records, vocab_size=vocab_n)
        print(f"  hijack layer={args.layer}: {len(records)} records", flush=True)

    elif args.task == "aggressive_sep":
        results = compute_aggressive_sep(
            model, block_size, vocab_n, DEVICE, args.layer,
            num_trials=args.num_trials,
        )
        intensities = np.array(sorted(results.keys()))
        rates = np.array([results[i] for i in intensities])
        np.savez(args.out, intensities=intensities, success_rates=rates)
        print(f"  aggressive_sep layer={args.layer}: done", flush=True)

    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
