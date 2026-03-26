#!/usr/bin/env python3
"""
Worker: compute average attention-by-number-value heatmap (256×256)
for both attention layers of each assigned checkpoint.

For each token position with value i, we accumulate its attention weights
to all visible positions with value j, then normalize by the total count
of from-positions with value i. The result: avg_matrix[i,j] ≈ fraction
of attention that number i pays to number j (rows sum to ~1).
"""
import argparse, json, os, sys, time, types
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'grid-run'))
from model_analysis import GPT, GPTConfig

VOCAB_SIZE = 256
BLOCK_SIZE = 16
SEQ_LEN = 2 * BLOCK_SIZE + 1  # 33
N_LAYERS = 2
BATCH_SIZE = 1024
N_BATCHES = 100  # 102 400 sequences total


def remap_state_dict(sd):
    new_sd = {}
    for key, val in sd.items():
        new_key = key
        for i in range(10):
            new_key = new_key.replace(f'transformer.h.{i}.attn.', f'transformer.h.{i}.c_attn.')
            new_key = new_key.replace(f'transformer.h.{i}.mlp.', f'transformer.h.{i}.c_fc.')
        new_sd[new_key] = val
    return new_sd


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    mc = ckpt['model_config']
    vocab_size = mc['vocab_size'] - 1
    block_size = mc['block_size']
    config = GPTConfig(block_size=block_size, vocab_size=vocab_size,
                       with_layer_norm=mc.get('use_final_LN', True))
    model = GPT(config)
    sd = remap_state_dict(ckpt['model_state_dict'])
    wpe_max = block_size * 4 + 1
    if 'transformer.wpe.weight' in sd and sd['transformer.wpe.weight'].shape[0] > wpe_max:
        sd['transformer.wpe.weight'] = sd['transformer.wpe.weight'][:wpe_max]
    for k in [k for k in sd if k.endswith('.c_attn.bias') and 'c_attn.c_attn' not in k]:
        del sd[k]
    sd.pop('lm_head.weight', None)
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    return model, config


def patch_attention(model):
    """Replace forward so it stores batched attention weights (B, 1, T, T)."""
    for layer_idx in range(N_LAYERS):
        attn_mod = model.transformer.h[layer_idx].c_attn

        def _make():
            def fwd(self_attn, x, layer_n=-1):
                B, T, C = x.size()
                qkv = self_attn.c_attn(x)
                q, k, v = qkv.split(self_attn.n_embd, dim=2)
                q = q.view(B, T, self_attn.n_heads, C // self_attn.n_heads).transpose(1, 2)
                k = k.view(B, T, self_attn.n_heads, C // self_attn.n_heads).transpose(1, 2)
                v = v.view(B, T, self_attn.n_heads, C // self_attn.n_heads).transpose(1, 2)
                a = q @ k.transpose(-1, -2) * 0.1 / k.size(-1) ** 0.5
                a = a.masked_fill(self_attn.bias[:, :, :T, :T] == 0, float('-inf'))
                a = F.softmax(a, dim=-1)
                self_attn.batched_attn = a
                y = (a @ v).transpose(1, 2).contiguous().view(B, T, C)
                return self_attn.c_proj(y)
            return fwd

        attn_mod.forward = types.MethodType(_make(), attn_mod)


def get_batch(device):
    ids = torch.rand(BATCH_SIZE, VOCAB_SIZE, device=device).argsort(dim=1)[:, :BLOCK_SIZE]
    sorted_ids, _ = ids.sort(dim=1)
    sep = torch.full((BATCH_SIZE, 1), VOCAB_SIZE, dtype=torch.long, device=device)
    return torch.cat([ids, sep, sorted_ids], dim=1)


@torch.no_grad()
def compute(model, device):
    T = SEQ_LEN
    VS = VOCAB_SIZE
    causal = torch.tril(torch.ones(T, T, device=device, dtype=torch.bool))

    sum_mat = [torch.zeros(VS * VS, device=device, dtype=torch.float64) for _ in range(N_LAYERS)]
    from_cnt = torch.zeros(VS, device=device, dtype=torch.float64)

    for _ in range(N_BATCHES):
        tokens = get_batch(device)
        model(tokens)

        from_v = tokens.unsqueeze(2).expand(-1, -1, T)
        to_v = tokens.unsqueeze(1).expand(-1, T, -1)
        valid = causal.unsqueeze(0) & (from_v < VS) & (to_v < VS)

        flat_idx = (from_v * VS + to_v).long()
        idx_v = flat_idx[valid]

        for layer in range(N_LAYERS):
            attn = model.transformer.h[layer].c_attn.batched_attn[:, 0]
            sum_mat[layer].scatter_add_(0, idx_v, attn[valid].double())

        tok_valid = tokens[tokens < VS]
        from_cnt.scatter_add_(0, tok_valid.long(),
                              torch.ones(tok_valid.numel(), device=device, dtype=torch.float64))

    results = []
    fc = from_cnt.clamp(min=1).unsqueeze(1)
    for layer in range(N_LAYERS):
        avg = (sum_mat[layer].view(VS, VS) / fc).cpu().numpy()
        results.append(avg)
    return results, from_cnt.cpu().numpy()


def plot_heatmap(avg, layer, out_dir, ckpt_label):
    fig, ax = plt.subplots(figsize=(10, 9))
    pos_vals = avg[avg > 0]
    vmax = np.percentile(pos_vals, 99) if pos_vals.size > 0 else 1.0
    im = ax.imshow(avg, aspect='auto', origin='lower', cmap='inferno',
                   vmin=0, vmax=vmax, interpolation='nearest')
    ax.set_xlabel('To number (attended-to)', fontsize=12)
    ax.set_ylabel('From number (attending)', fontsize=12)
    ax.set_title(f'Layer {layer+1}: avg attention  (number → number)\n{ckpt_label}',
                 fontsize=11)
    ticks = list(range(0, 256, 32)) + [255]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    fig.colorbar(im, ax=ax, shrink=0.82, label='Avg attention weight')
    plt.tight_layout()
    path = os.path.join(out_dir, f'avg_attn_by_number_layer{layer}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tasks-file', required=True)
    ap.add_argument('--gpu', type=int, required=True)
    args = ap.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = 'cuda'

    with open(args.tasks_file) as f:
        tasks = json.load(f)

    print(f"GPU {args.gpu}: {len(tasks)} checkpoints", flush=True)

    for task in tasks:
        ckpt_path = task['ckpt_path']
        out_dir = task['out_dir']
        label = os.path.basename(ckpt_path).replace('.pt', '')

        done0 = os.path.exists(os.path.join(out_dir, 'avg_attn_by_number_layer0.png'))
        done1 = os.path.exists(os.path.join(out_dir, 'avg_attn_by_number_layer1.png'))
        if done0 and done1:
            print(f"  Skip (exists): {label}", flush=True)
            continue

        t0 = time.time()
        model, _ = load_model(ckpt_path, device)
        patch_attention(model)
        avgs, from_cnt = compute(model, device)

        os.makedirs(out_dir, exist_ok=True)
        for layer in range(N_LAYERS):
            np.savez(os.path.join(out_dir, f'avg_attn_by_number_layer{layer}.npz'),
                     avg_attn=avgs[layer], from_count=from_cnt)
            plot_heatmap(avgs[layer], layer, out_dir, label)

        dt = time.time() - t0
        print(f"  Done: {label} ({dt:.1f}s)", flush=True)
        del model
        torch.cuda.empty_cache()

    print(f"GPU {args.gpu}: all done.", flush=True)


if __name__ == '__main__':
    main()
