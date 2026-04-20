#!/usr/bin/env python3
"""Quick plot showing the 2 outlier seeds vs the 23 normal seeds."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'sortgpt_toolkit'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from model import DEVICE, load_model_from_checkpoint

BASE = os.path.join(os.path.dirname(__file__), '..', '..')
OUTDIR = os.path.join(os.path.dirname(__file__), 'plots')
Z_REF, Y_REF = 250, 250

plt.rcParams.update({
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'savefig.facecolor': 'white', 'font.size': 11,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.alpha': 0.2,
})

def _ckpt(seed):
    if seed == 1:
        return os.path.join(BASE, 'new-grid', 'k32_N512', 'checkpoints',
                            'std0p01_iseed1__ckpt100000.pt')
    elif seed <= 5:
        return os.path.join(BASE, 'new-grid-multiple', 'k32_N512',
                            f'seed{seed}', 'checkpoints', f'std0p01_iseed{seed}__ckpt100000.pt')
    elif seed <= 15:
        return os.path.join(BASE, 'new-grid-multiple-2', 'k32_N512',
                            f'seed{seed}', 'checkpoints', f'std0p01_iseed{seed}__ckpt100000.pt')
    else:
        return os.path.join(BASE, 'new-grid-multiple-3', 'k32_N512',
                            f'seed{seed}', 'checkpoints', f'std0p01_iseed{seed}__ckpt100000.pt')

@torch.no_grad()
def compute_argmax_curve(ckpt_path):
    model = load_model_from_checkpoint(ckpt_path)
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
    inp_q = e_all[Z_REF].unsqueeze(0) + V_all
    Q = block1.ln_1(block0.mlp(block0.ln_2(inp_q))) @ W_q.T + b_q
    inp_k = e_all + V_all[Y_REF].unsqueeze(0)
    K = block1.ln_1(block0.mlp(block0.ln_2(inp_k))) @ W_k.T + b_k
    hm = (Q @ K.T).cpu().numpy()
    x_range = list(range(Z_REF, min(vocab_n, Z_REF + 260), 2))
    argmax_ts = [int(np.argmax(hm[x, :])) for x in x_range]
    del model; torch.cuda.empty_cache()
    return np.array(x_range), np.array(argmax_ts)

all_seeds = {}
for s in range(1, 26):
    print(f"seed {s}...", end=" ", flush=True)
    xr, am = compute_argmax_curve(_ckpt(s))
    all_seeds[s] = (xr, am)
    print(f"range [{am.min()}, {am.max()}]")

outliers = {8, 10}
x_range = all_seeds[1][0]

fig, ax = plt.subplots(figsize=(8, 5.5))
for s in range(1, 26):
    if s in outliers:
        continue
    ax.plot(x_range, all_seeds[s][1], linewidth=0.6, alpha=0.25, color='#4C72B0')

for s in sorted(outliers):
    ax.plot(x_range, all_seeds[s][1], linewidth=2, alpha=0.9,
            label=f'Seed {s} (outlier, range [{all_seeds[s][1].min()}, {all_seeds[s][1].max()}])')

ax.plot(x_range, x_range, 'k--', linewidth=0.8, alpha=0.3, label='$t = x$ (diagonal)')
ax.axhline(Z_REF, color='gray', linewidth=0.8, linestyle=':', alpha=0.5, label=f'$z = {Z_REF}$')
ax.set_xlabel('Query-side attn1 target ($x$)')
ax.set_ylabel('$\\mathrm{argmax}_t\\, \\mathrm{score}(x, t)$')
ax.set_title(f'Argmax Saturation: 23 normal seeds (blue) vs 2 outliers\n(k32_N512, $z = {Z_REF}$, $y = {Y_REF}$)')
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, 'argmax_saturation_outliers.png'), dpi=200, bbox_inches='tight')
plt.close()
print("Saved argmax_saturation_outliers.png")
