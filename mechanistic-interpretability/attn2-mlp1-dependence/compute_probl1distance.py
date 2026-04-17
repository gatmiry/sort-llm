"""
Compute L1 distance between normal and MLP1-only attn2 probability distributions.

For each leap-former checkpoint, measures how much attn2's attention pattern
changes when embed and attn1_out are removed from the residual stream,
leaving only mlp1_out. A small L1 distance means attn2 depends almost
exclusively on MLP1 output.

Usage:
    python compute_probl1distance.py --ckpt <path> --output <path> [--n-trials 15000]
"""
import sys, os, argparse, torch, numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'sortgpt_toolkit'))
from model import DEVICE, load_model_from_checkpoint, get_batch


def compute_probl1distance(model, n_trials=15000):
    bs = model.config.block_size
    vn = model.config.vocab_size - 1
    n_embd = model.config.n_embd
    b0, b1 = model.transformer.h[0], model.transformer.h[1]

    l1_distances = []
    for trial in range(n_trials):
        with torch.no_grad():
            idx = get_batch(1, bs, DEVICE, vocab_n=vn)
            B, T = idx.size()
            pos_emb = model.transformer.wpe(model.pos_idx[:T])
            embed = model.transformer.wte(idx) + pos_emb

            h0 = b0.ln_1(embed)
            qkv0 = b0.attn.c_attn(h0)
            q0, k0, v0 = qkv0.split(n_embd, dim=2)
            q0h = q0.view(B, T, 1, n_embd).transpose(1, 2)
            k0h = k0.view(B, T, 1, n_embd).transpose(1, 2)
            v0h = v0.view(B, T, 1, n_embd).transpose(1, 2)
            y0 = F.scaled_dot_product_attention(q0h, k0h, v0h, dropout_p=0.0, is_causal=True)
            attn1_out = b0.attn.c_proj(y0.transpose(1, 2).contiguous().view(B, T, n_embd))
            res_a1 = embed + attn1_out
            mlp1_out = b0.mlp(b0.ln_2(res_a1))

            res_normal = res_a1 + mlp1_out
            res_mlp1only = mlp1_out.clone()

            scale = n_embd ** 0.5
            causal_mask = torch.triu(torch.ones(T, T, device=DEVICE), diagonal=1).bool()

            def get_attn2_probs(residual):
                h1 = b1.ln_1(residual)
                qkv1 = b1.attn.c_attn(h1)
                q1, k1, _ = qkv1.split(n_embd, dim=2)
                q1h = q1.view(B, T, 1, n_embd).transpose(1, 2)
                k1h = k1.view(B, T, 1, n_embd).transpose(1, 2)
                scores = (q1h @ k1h.transpose(-2, -1)) / scale
                scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
                return F.softmax(scores, dim=-1).squeeze()

            probs_n = get_attn2_probs(res_normal)
            probs_m = get_attn2_probs(res_mlp1only)

            for p in range(bs - 1):
                qp = bs + 1 + p
                l1 = np.abs(probs_n[qp, :].cpu().numpy() - probs_m[qp, :].cpu().numpy()).sum()
                l1_distances.append(l1)

    return np.array(l1_distances)


def plot_probl1distance(l1_arr, title, outpath):
    mean_l1 = l1_arr.mean()
    median_l1 = np.median(l1_arr)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, log in zip(axes, [False, True]):
        ax.hist(l1_arr, bins=100, color='#2a9d8f', edgecolor='white', linewidth=0.3, density=True)
        ax.axvline(mean_l1, color='#e63946', linewidth=2, linestyle='--', label=f'Mean={mean_l1:.4f}')
        if not log:
            ax.axvline(median_l1, color='#264653', linewidth=2, linestyle=':', label=f'Median={median_l1:.4f}')
        ax.set_xlabel('L1 distance (sum |p_normal - p_mlp1only|)', fontsize=12)
        ax.set_ylabel('Density' + (' (log)' if log else ''), fontsize=12)
        ax.set_title(('Same (log scale)' if log else 'Normal vs mlp1-only attn2 probs'), fontsize=13)
        if log:
            ax.set_yscale('log')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--n-trials', type=int, default=15000)
    args = parser.parse_args()

    model = load_model_from_checkpoint(args.ckpt)
    l1_arr = compute_probl1distance(model, args.n_trials)

    print(f"Mean L1: {l1_arr.mean():.6f}")
    print(f"Median L1: {np.median(l1_arr):.6f}")
    print(f"P99: {np.percentile(l1_arr, 99):.6f}")
    print(f"Max: {l1_arr.max():.6f}")
    print(f"n: {len(l1_arr)}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    name = os.path.basename(args.ckpt).replace('.pt', '')
    plot_probl1distance(l1_arr, f'probl1distance: {name}', args.output)
    print(f"Saved: {args.output}")
