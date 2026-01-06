"""
Evaluate (1) whether attention at the *first sorted-output query position* peaks on the
correct input location, over 100 random sorting examples; and (2) if the peak location
is wrong, the average edit (Levenshtein) distance between the model’s predicted number
and the ground-truth number at that first query position.

Assumptions consistent with your get_batch():
- Sequence layout: [x (len=32), SEP (=vocab_size), sorted(x) (len=32)]
- "first query position" = the first token position in the sorted-output segment,
  i.e. position q_pos = block_size + 1 (0-indexed).
- "correct place" for attention = the position in the *input x segment* (0..31)
  that contains the ground-truth smallest element sorted(x)[0]. Since randperm gives
  unique numbers, that location is unique.

This script tries to extract attention weights via a forward hook on modules that look
like attention modules. In your code, attention is called as:
    model.transformer.h[0].c_attn(x, 0)
so we default to hooking model.transformer.h[*].c_attn.
If your attention weights are stored/returned differently, see the notes in
`capture_attention_hook`.
"""

import os
import torch

from model_tbyt_inference import GPT, GPTConfig

# ----------------------------
# Config / model load (yours)
# ----------------------------
itr_num = 100000
block_size = 32
vocab_size = 128
device = "cpu"

config = GPTConfig(block_size=block_size, vocab_size=vocab_size)
model = GPT(config)

ckpt_path = os.path.join(
    os.getcwd(),
    # 2 layer
    # f"saved_models/dec29-embedsize/dec29_tbyt_with-pos-embedding_n_embd:8_head:1_layers:2_vocab_size:128_itr:{itr_num}_checkpoint.pt",
    # Attention-argmax wrong count: 51 # Avg edit distance (pred vs GT) when attention is wrong: 1.235
    # 2 layer without mlp
    # f"saved_models/dec29-embedsize/dec29_tbyt_with-pos-embedding_mlp:False_n_embd:8_head:1_layers:2_vocab_size:128_itr:{itr_num}_checkpoint.pt",
    # Attention-argmax correct (first query position): 30/100 = 0.300
    # Attention-argmax wrong count: 70 # Avg edit distance (pred vs GT) when attention is wrong: 2.643
    # 1 layer
    f"saved_models/dec29-embedsize/dec29_tbyt_with-pos-embedding_n_embd:8_head:1_layers:1_vocab_size:128_itr:{itr_num}_checkpoint.pt",
    # Attention-argmax correct (first query position): 1/100 = 0.010
# Attention-argmax wrong count: 99
# Avg edit distance (pred vs GT) when attention is wrong: 1.929
    # 1 layer without mlp
    # f"saved_models/dec29-embedsize/dec29_tbyt_with-pos-embedding_mlp:False_n_embd:8_head:1_layers:1_vocab_size:128_itr:{itr_num}_checkpoint.pt",
    # 
)
state = torch.load(ckpt_path, map_location=device)
model.load_state_dict(state["model"])
model.to(device=device)
model.eval()

batch_size = 1

# ----------------------------
# Data: your get_batch()
# ----------------------------
def get_batch(changing_num=-1, changing_index=-1, initial_sequence=None, batch_size=batch_size):
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

    x = torch.stack([cat_sorted_tensor(torch.randperm(vocab_size)[:block_size]) for _ in range(batch_size)])
    return x

# ----------------------------
# Levenshtein edit distance
# ----------------------------
def edit_distance(a: str, b: str) -> int:
    # classic DP, O(len(a)*len(b)), fine for tiny strings like "127"
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
_attn_cache = []

def capture_attention_hook(module, inp, out):
    """
    Tries to cache attention weights in a fairly robust way.

    Common patterns:
    1) module returns (y, att) where att is [B, nH, T, T] or [B, T, T]
    2) module stores attention on module.att or module.attn after forward
    3) module returns only y, and attention is not accessible -> you'll need to
       modify the attention module to return/store weights.

    We append the *most recent* attention we can find per forward call.
    """
    att = None

    # Case 1: output is tuple/list containing attention
    if isinstance(out, (tuple, list)):
        # try to find a tensor that looks like attention
        for item in out[::-1]:
            if torch.is_tensor(item) and item.dim() in (3, 4):
                att = item
                break

    # Case 2: module stores it
    if att is None:
        for name in ("att", "attn", "attention", "weights", "att_weights"):
            if hasattr(module, name):
                cand = getattr(module, name)
                if torch.is_tensor(cand) and cand.dim() in (3, 4):
                    att = cand
                    break

    if att is not None:
        _attn_cache.append(att.detach().cpu())

def install_attention_hooks(model):
    """
    Prefer hooking blocks' c_attn modules (matches your manual call site).
    Falls back to hooking any module whose name contains 'attn' or 'attention'.
    """
    handles = []

    # Preferred: transformer blocks' c_attn
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        for bi, block in enumerate(model.transformer.h):
            if hasattr(block, "c_attn"):
                handles.append(block.c_attn.register_forward_hook(capture_attention_hook))

    # Fallback: names
    if not handles:
        for name, mod in model.named_modules():
            lname = name.lower()
            if "attn" in lname or "attention" in lname:
                handles.append(mod.register_forward_hook(capture_attention_hook))

    return handles

hook_handles = install_attention_hooks(model)

# ----------------------------
# Metric computation
# ----------------------------
@torch.no_grad()
def run_eval(n_examples=100):
    """
    Returns:
      attn_correct_count: number of examples where attention argmax hits the correct input location
      n: number of examples
      avg_edit_distance_when_attn_wrong: average edit distance between predicted token and ground truth
                                       for those examples where attention location is wrong
    """
    attn_correct = 0
    wrong_edit_dists = []

    # positions (0-indexed)
    x_start = 0
    x_end = block_size - 1          # inclusive
    sep_pos = block_size            # token == vocab_size
    q_pos = block_size + 1          # first sorted output position

    for _ in range(n_examples):
        idx = get_batch(batch_size=1).to(device)  # [1, 65]
        _attn_cache.clear()

        logits, loss = model(idx)  # logits: [B, T, vocab_size?]

        # --- ground truth for first query position ---
        x = idx[0, x_start:sep_pos]                         # [32]
        sorted_x = idx[0, sep_pos + 1 : sep_pos + 1 + 32]   # [32]
        gt_token = int(sorted_x[0].item())

        # correct input location in x
        correct_loc = int((x == gt_token).nonzero(as_tuple=False)[0].item())  # 0..31

        # --- model prediction at q_pos ---
        pred_token = int(torch.argmax(logits[0, q_pos], dim=-1).item())

        # --- attention check ---
        # We’ll use the *last captured attention* (typically last hooked layer)
        # and reduce heads if needed.
        if len(_attn_cache) == 0:
            raise RuntimeError(
                "No attention weights were captured.\n"
                "You likely need to modify the attention module (e.g., c_attn) to return "
                "or store attention weights so the hook can see them."
            )

        att = _attn_cache[-1]  # try the last layer

        # Normalize shape to [B, nH, T, T]
        if att.dim() == 3:
            att = att.unsqueeze(1)  # [B, 1, T, T]

        # Some implementations return [nH, B, T, T] or similar; attempt a minimal fix:
        if att.shape[0] != 1 and att.shape[1] == 1:
            # if it looks like [nH, B, T, T] with B=1
            if att.dim() == 4 and att.shape[1] == 1:
                att = att.permute(1, 0, 2, 3)  # -> [B, nH, T, T]

        if att.shape[0] != 1:
            raise RuntimeError(f"Unexpected attention batch dimension: att.shape={tuple(att.shape)}")

        # average over heads -> [T, T]
        att_mean = att[0].mean(dim=0)  # [T, T]

        # attention distribution for query position q_pos over keys 0..T-1
        row = att_mean[q_pos]  # [T]

        # restrict to input x segment keys 0..31
        row_x = row[x_start : sep_pos]  # [32]
        attn_argmax_loc = int(torch.argmax(row_x).item())  # 0..31

        if attn_argmax_loc == correct_loc:
            attn_correct += 1
        else:
            # edit distance between predicted token and GT token (as strings)
            wrong_edit_dists.append(edit_distance(str(pred_token), str(gt_token)))

    avg_ed = (sum(wrong_edit_dists) / len(wrong_edit_dists)) if wrong_edit_dists else 0.0
    return attn_correct, n_examples, avg_ed, len(wrong_edit_dists)

attn_correct_count, n, avg_edit_dist, n_wrong = run_eval(n_examples=100)

print(f"Attention-argmax correct (first query position): {attn_correct_count}/{n} = {attn_correct_count/n:.3f}")
print(f"Attention-argmax wrong count: {n_wrong}")
print(f"Avg edit distance (pred vs GT) when attention is wrong: {avg_edit_dist:.3f}")

# Cleanup hooks (optional)
for h in hook_handles:
    h.remove()