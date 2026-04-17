#!/usr/bin/env python3
"""
Generate argmax bias figure: how does the argmax of attn2 QK score relate
to x_min, x_mean, x_max for split attn1 with n = 1..4 tokens.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'sortgpt_toolkit'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from model import DEVICE, load_model_from_checkpoint

CKPT = os.path.join(os.path.dirname(__file__), '..', '..', 'new-grid', 'k32_N512',
                     'checkpoints', 'std0p01_iseed1__ckpt100000.pt')
OUTDIR = os.path.join(os.path.dirname(__file__), 'plots')
os.makedirs(OUTDIR, exist_ok=True)

plt.rcParams.update({'figure.facecolor': 'white', 'axes.facecolor': 'white',
                     'savefig.facecolor': 'white', 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': 0.2,
                     'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 13})

model = load_model_from_checkpoint(CKPT)
n_embd = model.config.n_embd
vocab_n = model.config.vocab_size - 1
block0, block1 = model.transformer.h[0], model.transformer.h[1]

e_all = model.transformer.wte.weight[:vocab_n]
ln1_e = block0.ln_1(e_all)
W_v = block0.attn.c_attn.weight[2*n_embd:3*n_embd, :]
b_v = block0.attn.c_attn.bias[2*n_embd:3*n_embd]
v = ln1_e @ W_v.T + b_v
V_all = v @ block0.attn.c_proj.weight.T + block0.attn.c_proj.bias

W_q = block1.attn.c_attn.weight[:n_embd, :]
b_q = block1.attn.c_attn.bias[:n_embd]
W_k = block1.attn.c_attn.weight[n_embd:2*n_embd, :]
b_k = block1.attn.c_attn.bias[n_embd:2*n_embd]


@torch.no_grad()
def get_argmax(z, query_tokens, key_y_tokens):
    V_q = torch.stack([V_all[t] for t in query_tokens]).mean(dim=0)
    inp_q = e_all[z].unsqueeze(0) + V_q.unsqueeze(0)
    mlp_q = block0.mlp(block0.ln_2(inp_q))
    q = (block1.ln_1(mlp_q) @ W_q.T + b_q).squeeze(0)

    V_k = torch.stack([V_all[t] for t in key_y_tokens]).mean(dim=0)
    inp_k = e_all + V_k.unsqueeze(0)
    mlp_k = block0.mlp(block0.ln_2(inp_k))
    K = block1.ln_1(mlp_k) @ W_k.T + b_k
    scores = (q @ K.T).cpu().numpy()
    return int(np.argmax(scores))


z_fixed = 250
y_tokens = [251]

# ---- Panel A: argmax vs x_min for different n, fixed delta ----
# ---- Panel B: (argmax - x_min) vs delta for different n, fixed x_min ----

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

# Panel A: argmax vs x_min, for n=1..4, delta=5
ax = axes[0]
x_min_range = np.arange(252, 400, 3)
colors_n = {1: '#1b7837', 2: '#2166ac', 3: '#d6604d', 4: '#762a83'}
delta_fixed = 5
for n_tok in [1, 2, 3, 4]:
    argmaxes = []
    valid_xmins = []
    for xm in x_min_range:
        tokens = [xm + i * delta_fixed for i in range(n_tok)]
        if max(tokens) >= vocab_n:
            break
        argmaxes.append(get_argmax(z_fixed, tokens, y_tokens))
        valid_xmins.append(xm)
    ax.plot(valid_xmins, argmaxes, color=colors_n[n_tok], linewidth=1.8,
            label=f'$n = {n_tok}$', alpha=0.85)

ax.plot([252, 400], [252, 400], 'k--', linewidth=0.8, alpha=0.4, label='$t^* = x_{\\min}$')
ax.set_xlabel('$x_{\\min}$')
ax.set_ylabel('Argmax key token ($t^*$)')
ax.set_title(f'Argmax vs $x_{{\\min}}$ ($\\delta = {delta_fixed}$)')
ax.legend(fontsize=9)

# Panel B: argmax vs x_min for different deltas, n=3
ax = axes[1]
n_fixed = 3
deltas_to_plot = [1, 5, 10, 20]
colors_d = {1: '#1b7837', 5: '#2166ac', 10: '#d6604d', 20: '#762a83'}
for delta in deltas_to_plot:
    argmaxes = []
    valid_xmins = []
    for xm in x_min_range:
        tokens = [xm + i * delta for i in range(n_fixed)]
        if max(tokens) >= vocab_n:
            break
        argmaxes.append(get_argmax(z_fixed, tokens, y_tokens))
        valid_xmins.append(xm)
    ax.plot(valid_xmins, argmaxes, color=colors_d[delta], linewidth=1.8,
            label=f'$\\delta = {delta}$', alpha=0.85)

ax.plot([252, 400], [252, 400], 'k--', linewidth=0.8, alpha=0.4, label='$t^* = x_{\\min}$')
ax.set_xlabel('$x_{\\min}$')
ax.set_ylabel('Argmax key token ($t^*$)')
ax.set_title(f'Argmax vs $x_{{\\min}}$ ($n = {n_fixed}$)')
ax.legend(fontsize=9)

# Panel C: (argmax - x_min) vs (argmax - x_mean) scatter for all configs
ax = axes[2]
for n_tok in [1, 2, 3, 4]:
    for delta in [1, 5, 10, 20]:
        diffs_min = []
        diffs_mean = []
        for xm in x_min_range:
            tokens = [xm + i * delta for i in range(n_tok)]
            if max(tokens) >= vocab_n:
                break
            am = get_argmax(z_fixed, tokens, y_tokens)
            diffs_min.append(am - xm)
            diffs_mean.append(am - np.mean(tokens))
        ax.scatter(diffs_min, diffs_mean, color=colors_n[n_tok],
                   alpha=0.15, s=12, edgecolors='none')

for n_tok in [1, 2, 3, 4]:
    all_diffs_min = []
    all_diffs_mean = []
    for delta in [1, 5, 10, 20]:
        for xm in x_min_range:
            tokens = [xm + i * delta for i in range(n_tok)]
            if max(tokens) >= vocab_n:
                break
            am = get_argmax(z_fixed, tokens, y_tokens)
            all_diffs_min.append(am - xm)
            all_diffs_mean.append(am - np.mean(tokens))
    ax.scatter(np.mean(all_diffs_min), np.mean(all_diffs_mean),
               color=colors_n[n_tok], s=120, edgecolors='black', linewidths=1.2,
               marker='D', zorder=5, label=f'$n = {n_tok}$ (mean)')

lims = [-150, 30]
ax.plot(lims, lims, 'k--', linewidth=0.8, alpha=0.4)
ax.set_xlabel('$t^* - x_{\\min}$')
ax.set_ylabel('$t^* - x_{\\mathrm{mean}}$')
ax.set_title('Argmax offset from $x_{\\min}$ vs $x_{\\mathrm{mean}}$')
ax.legend(fontsize=9)

fig.suptitle('Argmax Bias Analysis: Attn2 Argmax Tracks $x_{\\min}$, Not $x_{\\mathrm{mean}}$',
             fontsize=14, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.91])
out = os.path.join(OUTDIR, 'argmax_bias_analysis.png')
fig.savefig(out, dpi=200, bbox_inches='tight')
plt.close()
print(f"Saved {out}")
