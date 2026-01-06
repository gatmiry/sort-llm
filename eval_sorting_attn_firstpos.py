#!/usr/bin/env python3
"""
eval_sorting_attn_firstpos.py

Loads 4 models, runs 100 sorting examples, and evaluates:
- attention-hit-rate: for the first sorted-output query position, does attn argmax land on the correct input position?
- if attention is wrong: average Levenshtein edit distance between predicted number and ground-truth number (first sorted token)
"""

import os
import argparse
import torch
import torch.nn as nn

from model_tbyt_inference import GPT, GPTConfig


# -------------------------
# Task / batch generation
# -------------------------
def get_batch(vocab_size: int, changing_num=-1, changing_index=-1, initial_sequence=None, batch_size=1, unsorted_len=32):
    """
    Produces a single sequence:
      [unsorted (unsorted_len)] [SEP=vocab_size] [sorted (unsorted_len)]
    Total length = 2*unsorted_len + 1
    """
    def cat_sorted_tensor(x):
        if initial_sequence is not None:
            x = initial_sequence.clone()
        else:
            x = x

        if changing_num != -1:
            if changing_index == -1:
                x[0] = changing_num
            else:
                x[changing_index] = changing_num

        vals, _ = torch.sort(x)
        return torch.cat((x, torch.tensor([vocab_size], dtype=x.dtype), vals), dim=0)

    x = torch.stack([cat_sorted_tensor(torch.randperm(vocab_size)[:unsorted_len]) for _ in range(batch_size)])
    return x


# -------------------------
# Edit distance (Levenshtein)
# -------------------------
def levenshtein(a: str, b: str) -> int:
    # classic DP
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            ins = cur[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            cur.append(min(ins, dele, sub))
        prev = cur
    return prev[-1]


# -------------------------
# Disable MLP (best-effort)
# -------------------------
class ZeroModule(nn.Module):
    """Returns zeros of the same shape as input (so residual add does nothing)."""
    def forward(self, x, *args, **kwargs):
        return torch.zeros_like(x)

def disable_mlp_inplace(model: nn.Module):
    """
    Best-effort: replaces common MLP modules with "zero" modules so residual path is unchanged.
    Adjust this if your code uses different names.
    """
    # common nanoGPT-like naming: block.mlp, block.c_fc, block.c_proj
    for blk in getattr(model, "transformer", {}).h:
        if hasattr(blk, "mlp"):
            blk.mlp = ZeroModule()
        if hasattr(blk, "c_fc"):
            blk.c_fc = ZeroModule()
        if hasattr(blk, "c_proj") and isinstance(getattr(blk, "c_proj"), nn.Module):
            # if c_proj is part of mlp in your impl; safe to leave alone if unsure
            pass
    return model


# -------------------------
# Attention capture via hooks (best-effort)
# -------------------------
class AttnCatcher:
    """
    Captures attention weights from a forward hook.
    We support a few common patterns:
      - module returns (y, att) or (y, att, ...)
      - module sets module.att or module.last_att
      - module returns dict with 'att' / 'attn'
    """
    def __init__(self):
        self.last_att = None

    def hook(self, module, inputs, outputs):
        att = None

        # 1) outputs is tuple/list and contains an attention tensor
        if isinstance(outputs, (tuple, list)):
            for item in outputs:
                if torch.is_tensor(item) and item.dim() in (3, 4):
                    # likely (B, nh, T, T) or (B, T, T)
                    att = item
                    break

        # 2) dict-like
        if att is None and isinstance(outputs, dict):
            for k in ("att", "attn", "attention"):
                if k in outputs and torch.is_tensor(outputs[k]):
                    att = outputs[k]
                    break

        # 3) module attribute
        if att is None:
            for k in ("att", "attn", "last_att", "last_attn", "attention"):
                if hasattr(module, k):
                    v = getattr(module, k)
                    if torch.is_tensor(v) and v.dim() in (3, 4):
                        att = v
                        break

        self.last_att = att


def find_attention_module(model: nn.Module, prefer_layer: int = 0):
    """
    Tries to locate an attention submodule to hook.
    Common possibilities:
      - model.transformer.h[i].attn
      - model.transformer.h[i].c_attn
      - model.transformer.h[i].self_attn
    Returns (module, description_string).
    """
    blocks = getattr(getattr(model, "transformer", None), "h", None)
    if blocks is None:
        raise RuntimeError("Could not find model.transformer.h (transformer blocks). Adjust find_attention_module().")

    i = min(prefer_layer, len(blocks) - 1)
    blk = blocks[i]

    for name in ("attn", "self_attn", "c_attn"):
        if hasattr(blk, name) and isinstance(getattr(blk, name), nn.Module):
            return getattr(blk, name), f"transformer.h[{i}].{name}"

    # fallback: search any submodule with "att" in name
    for subname, submod in blk.named_modules():
        if "att" in subname.lower():
            return submod, f"transformer.h[{i}].{subname}"

    raise RuntimeError(
        "Could not find an attention module to hook inside transformer.h[0]. "
        "Update find_attention_module() with the correct attribute name from your Block."
    )


# -------------------------
# Model loading (robust)
# -------------------------
def load_model_from_ckpt(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device)

    # If your checkpoint includes config, prefer it.
    # Otherwise fall back to defaults; we will patch block_size after we see batch length.
    if isinstance(ckpt, dict) and "config" in ckpt:
        cfg = GPTConfig(**ckpt["config"])
    else:
        # default fallback; will override block_size later
        cfg = GPTConfig(block_size=32, vocab_size=128)

    model = GPT(cfg)

    # your checkpoints look like {'model': state_dict}
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model, cfg


# -------------------------
# Evaluation
# -------------------------
@torch.no_grad()
def evaluate_one_model(
    model: nn.Module,
    vocab_size: int,
    num_examples: int = 100,
    device: str = "cpu",
    unsorted_len: int = 32,
    hook_layer: int = 0,
):
    """
    For each example:
      - query position = first sorted token position (unsorted_len + 1)  [0-index]
      - correct key position = argmin in unsorted prefix [0..unsorted_len-1]
      - predicted number = argmax(logits at sep_pos) where sep_pos = unsorted_len
      - if attn misses: record levenshtein(str(pred), str(gt_first_sorted))
    """
    # attach hook
    attn_mod, attn_desc = find_attention_module(model, prefer_layer=hook_layer)
    catcher = AttnCatcher()
    handle = attn_mod.register_forward_hook(catcher.hook)

    hits = 0
    miss_edit_dists = []

    sep_pos = unsorted_len               # position of [SEP]
    first_sorted_pos = unsorted_len + 1  # position of first sorted token
    # logits at position t predict token at t+1 (standard GPT)
    pred_pos_for_first_sorted = sep_pos

    for _ in range(num_examples):
        idx = get_batch(vocab_size=vocab_size, batch_size=1, unsorted_len=unsorted_len).to(device)
        T = idx.size(1)

        # If model was built with too-small block_size, this will crash.
        logits, _loss = model(idx)

        att = catcher.last_att
        if att is None:
            handle.remove()
            raise RuntimeError(
                f"Hooked {attn_desc} but did not capture attention weights. "
                "Your attention module likely doesn't return/store attn in a detectable way.\n"
                "Fix: modify AttnCatcher.hook() or your attention module to expose weights."
            )

        # att shape: (B, nh, T, T) or (B, T, T)
        if att.dim() == 4:
            # average over heads
            att_row = att[0].mean(dim=0)[first_sorted_pos]  # (T,)
        elif att.dim() == 3:
            att_row = att[0][first_sorted_pos]             # (T,)
        else:
            handle.remove()
            raise RuntimeError(f"Unexpected attention tensor shape: {tuple(att.shape)}")

        # we care about attention to unsorted prefix keys [0..unsorted_len-1]
        att_to_unsorted = att_row[:unsorted_len]
        pred_key_pos = int(torch.argmax(att_to_unsorted).item())

        unsorted_prefix = idx[0, :unsorted_len]
        correct_key_pos = int(torch.argmin(unsorted_prefix).item())

        if pred_key_pos == correct_key_pos:
            hits += 1
        else:
            # predicted first sorted number via logits at sep_pos
            pred_token = int(torch.argmax(logits[0, pred_pos_for_first_sorted]).item())
            gt_token = int(idx[0, first_sorted_pos].item())
            miss_edit_dists.append(levenshtein(str(pred_token), str(gt_token)))

    handle.remove()

    hit_rate = hits / num_examples
    avg_edit = (sum(miss_edit_dists) / len(miss_edit_dists)) if miss_edit_dists else 0.0

    return {
        "hit_rate": hit_rate,
        "num_hits": hits,
        "num_examples": num_examples,
        "num_misses": num_examples - hits,
        "avg_edit_distance_on_misses": avg_edit,
        "hooked_attention_module": attn_desc,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num_examples", type=int, default=100)
    parser.add_argument("--vocab_size", type=int, default=128)
    parser.add_argument("--unsorted_len", type=int, default=32)
    parser.add_argument("--hook_layer", type=int, default=0)

    # Provide your 4 checkpoint paths here (or pass via CLI)
    parser.add_argument("--ckpt_2layer", type=str, default="saved_models/dec29-embedsize/dec29_tbyt_with-pos-embedding_n_embd:8_head:1_layers:2_vocab_size:128_itr:100000_checkpoint.pt")
    parser.add_argument("--ckpt_2layer_nomlp", type=str, default="saved_models/dec29-embedsize/dec29_tbyt_with-pos-embedding_mlp:False_n_embd:8_head:1_layers:2_vocab_size:128_itr:60000_checkpoint.pt")
    parser.add_argument("--ckpt_1layer", type=str, default="saved_models/dec29-embedsize/dec29_tbyt_with-pos-embedding_n_embd:8_head:1_layers:1_vocab_size:128_itr:100000_checkpoint.pt")
    parser.add_argument("--ckpt_1layer_nomlp", type=str, default="saved_models/dec29-embedsize/dec29_tbyt_with-pos-embedding_mlp:False_n_embd:8_head:1_layers:1_vocab_size:128_itr:100000_checkpoint.pt")

    args = parser.parse_args()

    device = args.device
    vocab_size = args.vocab_size

    # Make a batch once so we know the true sequence length and can set block_size if needed
    probe = get_batch(vocab_size=vocab_size, batch_size=1, unsorted_len=args.unsorted_len)
    true_T = probe.size(1)

    def load_and_patch(ckpt_path: str):
        model, cfg = load_model_from_ckpt(ckpt_path, device=device)

        # If config.block_size is too small for the actual sequence length, patch it.
        # Many GPT impls only use block_size for positional embeddings / causal mask size.
        if hasattr(cfg, "block_size") and cfg.block_size < true_T:
            # patch model/config in-place (common in nanoGPT-style code)
            cfg.block_size = true_T
            if hasattr(model, "config"):
                model.config.block_size = true_T
            # If you have wpe sized to old block_size, this will still fail.
            # In that case your checkpoint/config must match training; set correct block_size there.
        return model

    # Load models
    m2 = load_and_patch(args.ckpt_2layer)
    m2_nomlp = load_and_patch(args.ckpt_2layer_nomlp)
    m1 = load_and_patch(args.ckpt_1layer)
    m1_nomlp = load_and_patch(args.ckpt_1layer_nomlp)

    # If your "*_nomlp" checkpoints are actually normal checkpoints and you want to disable MLP at eval time:
    # uncomment the next two lines and remove ckpt_2layer_nomlp/ckpt_1layer_nomlp usage.
    # m2_nomlp = disable_mlp_inplace(m2_nomlp)
    # m1_nomlp = disable_mlp_inplace(m1_nomlp)

    models = [
        ("2-layer", m2),
        ("2-layer no-MLP", m2_nomlp),
        ("1-layer", m1),
        ("1-layer no-MLP", m1_nomlp),
    ]

    print(f"Sequence length T = {true_T} (unsorted_len={args.unsorted_len}, sep=1, sorted_len={args.unsorted_len})")
    print(f"Evaluating num_examples={args.num_examples}, device={device}, hook_layer={args.hook_layer}\n")

    for name, model in models:
        stats = evaluate_one_model(
            model=model,
            vocab_size=vocab_size,
            num_examples=args.num_examples,
            device=device,
            unsorted_len=args.unsorted_len,
            hook_layer=args.hook_layer,
        )
        print(f"== {name} ==")
        print(f"Hooked: {stats['hooked_attention_module']}")
        print(f"Attention hit rate (first sorted pos): {stats['num_hits']}/{stats['num_examples']} = {stats['hit_rate']:.3f}")
        print(f"Avg edit distance on attention-misses: {stats['avg_edit_distance_on_misses']:.3f}  (misses={stats['num_misses']})")
        print()

if __name__ == "__main__":
    main()