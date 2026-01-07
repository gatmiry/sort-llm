




#!/usr/bin/env python3
"""
Eval script (teacher-forced) for your sorting model that reports:

(1) Attention-argmax correctness for predicting the *first sorted output token*
    - Autoregressive alignment: sorted_x[0] is at position out_start=33
      and is predicted by logits at position sep_pos=32 (SEP).
    - We therefore inspect attention row at query position q_pos = sep_pos (32),
      and check whether its argmax over the x-segment (keys 0..31) hits the
      true location of the minimum element.

(2) "After-MLP" prediction quality at the same autoregressive step:
    - We do NOT do influence/ablation.
    - We simply check whether the predicted first output token is correct
      (from logits at sep_pos), and if wrong compute edit distance between
      pred_first and gt_first (as strings).

Also reports:
- avg token accuracy over the full sorted-output segment (32 tokens)
- exact-match (all 32 correct)

NOTES (important for your codebase):
- Your model's CasualSelfAttention currently sets self.attn only when layer_n == 0.
  If you have n_layers > 1 and want last-layer attention, modify your model to set
  `self.attn = attn` for every layer after softmax. For n_layers=1, you're fine.

- This script uses teacher-forced logits, aligned with your training loss slicing.

Run:
  python eval_sort_first_token_attn_and_pred.py
"""

import os
import torch

from model_tbyt_inference import GPT, GPTConfig  # adjust if your import path differs


# ----------------------------
# Config / model load (yours)
# ----------------------------
itr_num = 80000
block_size = 32
vocab_size = 128
device = "cpu"

N_EXAMPLES = 100
BATCH_SIZE = 1

# Which attention layer to use:
# - "last" means: take highest layer index captured
# - int means: pick that layer if present, else fallback to highest captured
WHICH_LAYER = "last"

config = GPTConfig(block_size=block_size, vocab_size=vocab_size)
model = GPT(config)

ckpt_path = os.path.join(
    os.getcwd(),
    # 2 layer
    f"saved_models/dec29-embedsize/dec29_tbyt_with-pos-embedding_n_embd:8_head:1_layers:2_vocab_size:128_itr:{itr_num}_checkpoint.pt",
    # Attention-argmax wrong count: 51 # Avg edit distance (pred vs GT) when attention is wrong: 1.235 Avg token accuracy over full sorted-output (32 toks): 0.047
    # f"saved_models/dec29-embedsize/dec29_tbyt_with-pos-embedding_mlp:True_n_embd:64_head:1_layers:2_vocab_size:128_itr:{itr_num}_checkpoint.pt",
    # 2 layer without mlp
    # f"saved_models/dec29-embedsize/dec29_tbyt_with-pos-embedding_mlp:False_n_embd:8_head:1_layers:2_vocab_size:128_itr:{itr_num}_checkpoint.pt",
    # Attention-argmax correct (first query position): 30/100 = 0.300
    # Attention-argmax wrong count: 70 # Avg edit distance (pred vs GT) when attention is wrong: 2.643 Avg token accuracy over full sorted-output (32 toks): 0.018
    # 1 layer
    # f"saved_models/dec29-embedsize/dec29_tbyt_with-pos-embedding_n_embd:8_head:1_layers:1_vocab_size:128_itr:{itr_num}_checkpoint.pt",
    # Attention-argmax correct (first query position): 1/100 = 0.010 # Attention-argmax wrong count: 99 # Avg edit distance (pred vs GT) when attention is wrong: 1.929 Avg token accuracy over full sorted-output (32 toks): 0.019
    # 1 layer without mlp
    # f"saved_models/dec29-embedsize/dec29_tbyt_with-pos-embedding_mlp:False_n_embd:8_head:1_layers:1_vocab_size:128_itr:{itr_num}_checkpoint.pt",
    # 
)
state = torch.load(ckpt_path, map_location=device)
model.load_state_dict(state["model"])
model.to(device=device)
model.eval()


# ----------------------------
# Data: your get_batch()
# ----------------------------
def get_batch(changing_num=-1, changing_index=-1, initial_sequence=None, batch_size=BATCH_SIZE):
    def cat_sorted_tensor(x):
        if initial_sequence is not None:
            x = initial_sequence
        if changing_num != -1:
            if changing_index == -1:
                x[0] = changing_num
            else:
                x[changing_index] = changing_num
        vals, _ = torch.sort(x)
        return torch.cat((x, torch.tensor([vocab_size]), vals), dim=0)

    x = torch.stack(
        [cat_sorted_tensor(torch.randperm(vocab_size)[:block_size]) for _ in range(batch_size)]
    )
    return x


# ----------------------------
# Levenshtein edit distance
# ----------------------------
def edit_distance(a: str, b: str) -> int:
    n, m = len(a), len(b)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            cur = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(
                dp[j] + 1,      # deletion
                dp[j - 1] + 1,  # insertion
                prev + cost     # substitution
            )
            prev = cur
    return dp[m]


# ----------------------------
# Attention capture (hooks)
# ----------------------------
class AttnRecorder:
    """
    Record (layer_idx, attn) from each block.c_attn (your CasualSelfAttention).
    We read module.attn if present.
    """

    def __init__(self):
        self.records = []

    def clear(self):
        self.records.clear()

    def make_hook(self, layer_idx: int):
        def _hook(module, inp, out):
            att = None
            if hasattr(module, "attn") and torch.is_tensor(module.attn) and module.attn.dim() in (3, 4):
                att = module.attn
            elif hasattr(module, "att") and torch.is_tensor(module.att) and module.att.dim() in (3, 4):
                att = module.att

            if att is not None:
                self.records.append((layer_idx, att.detach().cpu()))
        return _hook


def install_attention_hooks(model: torch.nn.Module, recorder: AttnRecorder):
    handles = []
    if not (hasattr(model, "transformer") and hasattr(model.transformer, "h")):
        raise RuntimeError("Model does not have model.transformer.h; adjust hook installation.")

    for li, block in enumerate(model.transformer.h):
        if not hasattr(block, "c_attn"):
            raise RuntimeError(f"Block {li} has no c_attn; adjust hook installation.")
        handles.append(block.c_attn.register_forward_hook(recorder.make_hook(li)))
    return handles


def select_attention(records, which_layer):
    if not records:
        return None

    if which_layer == "last":
        best_li = max(li for li, _ in records)
        for li, att in reversed(records):
            if li == best_li:
                return (li, att)

    if isinstance(which_layer, int):
        for li, att in reversed(records):
            if li == which_layer:
                return (li, att)
        # fallback
        best_li = max(li for li, _ in records)
        for li, att in reversed(records):
            if li == best_li:
                return (li, att)

    # default fallback
    best_li = max(li for li, _ in records)
    for li, att in reversed(records):
        if li == best_li:
            return (li, att)
    return None


# ----------------------------
# Eval
# ----------------------------
@torch.no_grad()
def run_eval(model, n_examples=100, which_layer="last", return_per_example=False):
    recorder = AttnRecorder()
    handles = install_attention_hooks(model, recorder)

    # layout: [x (0..31), SEP (32), sorted(x) (33..64)]
    x_start = 0
    sep_pos = block_size
    out_start = sep_pos + 1
    out_end = out_start + block_size  # exclusive

    # AR alignment: logits at sep_pos predict token at out_start (sorted_x[0])
    q_pos = sep_pos

    # metrics
    attn_correct_count = 0
    wrong_attn_edit_dists = []

    first_pred_correct_count = 0
    wrong_first_pred_edit_dists = []

    token_accs = []
    seq_accs = []

    per_ex = []

    for ex_i in range(n_examples):
        idx = get_batch(batch_size=1).to(device)  # [1, 65]
        recorder.clear()

        logits, loss = model(idx)  # logits [1, T, vocab_size]

        # segments
        x = idx[0, x_start:sep_pos]                 # [32]
        sorted_x = idx[0, out_start:out_end]        # [32]

        gt_first = int(sorted_x[0].item())
        correct_loc = int((x == gt_first).nonzero(as_tuple=False)[0].item())  # 0..31

        # teacher-forced predictions aligned to training:
        # logits positions 32..63 predict tokens 33..64
        pred_sorted = logits[0, sep_pos:sep_pos + block_size, :].argmax(dim=-1)  # [32]

        token_correct = (pred_sorted == sorted_x).float()
        token_acc = float(token_correct.mean().item())
        seq_acc = float(token_correct.all().item())

        token_accs.append(token_acc)
        seq_accs.append(seq_acc)

        pred_first = int(pred_sorted[0].item())

        # (2) AFTER-MLP: correctness + edit distance for first predicted output token
        first_pred_correct = (pred_first == gt_first)
        if first_pred_correct:
            first_pred_correct_count += 1
            first_pred_ed = None
        else:
            first_pred_ed = edit_distance(str(pred_first), str(gt_first))
            wrong_first_pred_edit_dists.append(first_pred_ed)

        # (1) attention-argmax correctness at q_pos=SEP over keys in x-segment
        picked = select_attention(recorder.records, which_layer)
        if picked is None:
            for h in handles:
                h.remove()
            raise RuntimeError(
                "No attention weights captured.\n"
                "If n_layers>1 and you want last-layer attention, modify CasualSelfAttention.forward to always set self.attn = attn.\n"
                "For n_layers=1, ensure your forward actually assigns self.attn before returning."
            )

        used_layer, att = picked  # att on CPU
        # normalize to [B, nH, T, T]
        if att.dim() == 3:
            att = att.unsqueeze(1)

        if att.shape[0] != 1:
            for h in handles:
                h.remove()
            raise RuntimeError(f"Unexpected attention batch dim: {att.shape}")

        att_mean = att[0].mean(dim=0)   # [T, T]
        row = att_mean[q_pos]           # [T]
        row_x = row[x_start:sep_pos]    # [32]
        attn_argmax_loc = int(torch.argmax(row_x).item())
        attn_first_correct = (attn_argmax_loc == correct_loc)

        if attn_first_correct:
            attn_correct_count += 1
            wrong_attn_ed = None
        else:
            wrong_attn_ed = edit_distance(str(pred_first), str(gt_first))
            wrong_attn_edit_dists.append(wrong_attn_ed)

        if return_per_example:
            per_ex.append({
                "example": ex_i,
                "used_layer": used_layer,
                "gt_first": gt_first,
                "pred_first": pred_first,
                "first_pred_correct": first_pred_correct,
                "first_pred_edit_dist_if_wrong": first_pred_ed,
                "attn_first_correct": attn_first_correct,
                "attn_first_argmax_loc": attn_argmax_loc,
                "attn_first_correct_loc": correct_loc,
                "token_acc": token_acc,
                "seq_acc": seq_acc,
            })

    # aggregates
    avg_wrong_attn_ed = (sum(wrong_attn_edit_dists) / len(wrong_attn_edit_dists)) if wrong_attn_edit_dists else 0.0
    avg_wrong_firstpred_ed = (sum(wrong_first_pred_edit_dists) / len(wrong_first_pred_edit_dists)) if wrong_first_pred_edit_dists else 0.0

    results = {
        "n": n_examples,
        "which_layer": which_layer,

        # attention metric
        "attn_correct_count": attn_correct_count,
        "attn_wrong_count": len(wrong_attn_edit_dists),
        "avg_edit_dist_when_attn_wrong": avg_wrong_attn_ed,

        # after-MLP prediction metric (no influence)
        "first_pred_correct_count": first_pred_correct_count,
        "first_pred_wrong_count": len(wrong_first_pred_edit_dists),
        "avg_edit_dist_when_first_pred_wrong": avg_wrong_firstpred_ed,

        # overall sorting quality
        "avg_token_acc": sum(token_accs) / len(token_accs) if token_accs else 0.0,
        "seq_acc_rate": sum(seq_accs) / len(seq_accs) if seq_accs else 0.0,
    }
    if return_per_example:
        results["per_example"] = per_ex

    for h in handles:
        h.remove()

    return results


if __name__ == "__main__":
    results = run_eval(model, n_examples=N_EXAMPLES, which_layer=WHICH_LAYER, return_per_example=False)

    print(f"Layer used (requested): {results['which_layer']}")
    print()
    print("=== (1) Attention alignment for predicting sorted_x[0] ===")
    print(
        f"Attention-argmax correct (query=SEP pos 32, keys=x[0..31]): "
        f"{results['attn_correct_count']}/{results['n']} = {results['attn_correct_count']/results['n']:.3f}"
    )
    print(f"Attention-argmax wrong count: {results['attn_wrong_count']}")
    print(f"Avg edit distance (pred_first vs gt_first) when attention is wrong: {results['avg_edit_dist_when_attn_wrong']:.3f}")
    print()
    print("=== (2) After-MLP prediction quality at first output token ===")
    print(
        f"First output token correct (pred from logits at SEP pos 32): "
        f"{results['first_pred_correct_count']}/{results['n']} = {results['first_pred_correct_count']/results['n']:.3f}"
    )
    print(f"First output token wrong count: {results['first_pred_wrong_count']}")
    print(f"Avg edit distance (pred_first vs gt_first) when first pred is wrong: {results['avg_edit_dist_when_first_pred_wrong']:.3f}")
    print()
    print("=== Overall sorting quality (teacher-forced) ===")
    print(f"Avg token accuracy over full sorted-output (32 toks): {results['avg_token_acc']:.3f}")
    print(f"Sequence exact-match sorting accuracy (all 32 correct): {results['seq_acc_rate']:.3f}")