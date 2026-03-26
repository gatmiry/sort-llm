"""
Batch script to generate indicative match plots and length generalization plots
for all configs in grid-run/outputs.
"""
import os, sys, re, glob, time
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model_analysis import GPT, GPTConfig

DEVICE = 'cuda'
NUM_INDICATIVE_TRIALS = 1000
LEN_GEN_BATCHES = 10
LEN_GEN_BATCH_SIZE = 64
EVAL_LENGTHS = [4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64]
POS_OFFSET = 5

OUTPUTS = 'outputs'

def load_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    cfg = ckpt['config']
    config = GPTConfig(block_size=cfg['block_size'], vocab_size=cfg['vocab_size'],
                       with_layer_norm=cfg['with_layer_norm'])
    model = GPT(config)
    model.load_state_dict(ckpt['model'])
    model.to(DEVICE).eval()
    return model, config

def get_batch_single(vs, bs):
    x = torch.randperm(vs)[:bs]
    vals, _ = torch.sort(x)
    return torch.cat((x, torch.tensor([vs]), vals), dim=0).unsqueeze(0).to(DEVICE)

def get_batch_multi(batch_size, length, vs):
    seqs = []
    for _ in range(batch_size):
        x = torch.randperm(vs)[:length]
        vals, _ = torch.sort(x)
        seqs.append(torch.cat((x, torch.tensor([vs]), vals)))
    return torch.stack(seqs).to(DEVICE)

comp_names = ['L0_attn', 'L0_mlp', 'L1_attn', 'L1_mlp']

@torch.no_grad()
def run_indicative(model, config, num_trials):
    bs = config.block_size; vs = config.vocab_size
    wte = model.transformer.wte.weight[:vs]
    match_logit = np.zeros((bs, 4)); match_correct = np.zeros((bs, 4)); total_per_pos = np.zeros(bs)

    for t in range(num_trials):
        idx = get_batch_single(vs, bs)
        emb = model.transformer.wte(idx); b0 = model.transformer.h[0]
        a0 = b0.c_attn(b0.ln_1(emb)); x0 = emb + a0
        m0 = b0.c_fc(b0.ln_2(x0)); x0 = x0 + m0
        b1 = model.transformer.h[1]
        a1 = b1.c_attn(b1.ln_1(x0)); x1 = x0 + a1
        m1 = b1.c_fc(b1.ln_2(x1)); x1 = x1 + m1
        xf = model.transformer.ln_f(x1) if config.with_layer_norm else x1
        logits = model.lm_head(xf)

        for po in range(bs):
            qp = bs + po
            if qp + 1 >= idx.shape[1]: continue
            correct_next = idx[0, qp + 1].item()
            model_pred = logits[0, qp, :vs].argmax().item()
            comps = [a0[0, qp], m0[0, qp], a1[0, qp], m1[0, qp]]
            for c, cv in enumerate(comps):
                ind = (cv @ wte.T).argmax().item()
                if ind == model_pred: match_logit[po, c] += 1
                if ind == correct_next: match_correct[po, c] += 1
            total_per_pos[po] += 1

    ml_pct = 100 * match_logit / np.maximum(total_per_pos[:, None], 1)
    mc_pct = 100 * match_correct / np.maximum(total_per_pos[:, None], 1)
    total_all = total_per_pos.sum()
    ml_all = 100 * match_logit.sum(axis=0) / max(total_all, 1)
    mc_all = 100 * match_correct.sum(axis=0) / max(total_all, 1)
    return ml_pct, mc_pct, ml_all, mc_all

def _patched_attn_forward(self, x, layer_n=-1):
    """Forward that works with any batch size (skips raw_attn storage)."""
    B, T, C = x.size()
    qkv = self.c_attn(x)
    q, k, v = qkv.split(self.n_embd, dim=2)
    q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
    k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
    v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
    y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    y = y.transpose(1, 2).contiguous().view(B, T, C)
    y = self.c_proj(y)
    return y

@torch.no_grad()
def run_length_gen(model, config):
    vs = config.vocab_size; bs = config.block_size
    # Monkey-patch attention to handle batch_size > 1
    import types
    originals = []
    for block in model.transformer.h:
        originals.append(block.c_attn.forward)
        block.c_attn.forward = types.MethodType(_patched_attn_forward, block.c_attn)

    results = []
    for length in EVAL_LENGTHS:
        if length > vs:
            results.append(0.0)
            continue
        tok_correct = tok_total = 0
        for _ in range(LEN_GEN_BATCHES):
            actual_len = min(length, vs)
            idx = get_batch_multi(LEN_GEN_BATCH_SIZE, actual_len, vs)
            logits, _ = model(idx)
            targets = idx[:, actual_len + 1:]
            preds = logits[:, actual_len:2*actual_len, :vs].argmax(dim=-1)
            tok_correct += int((preds == targets).sum().item())
            tok_total += int(targets.numel())
        results.append(tok_correct / max(tok_total, 1))

    # Restore original forwards
    for block, orig in zip(model.transformer.h, originals):
        block.c_attn.forward = orig

    return results

def plot_indicative(out_dir, cfg_name, results):
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for metric, ylab, fname in [
        ('ml_pct', '% indicative matches model prediction', 'indicative_match_logit'),
        ('mc_pct', '% indicative matches correct next token', 'indicative_match_correct'),
    ]:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
        for i, (lt, tt) in enumerate([('LN0','Without LayerNorm'),('LN1','With LayerNorm')]):
            if lt not in results: continue
            ax = axes[i]; r = results[lt]; bs = r['ml_pct'].shape[0]
            for c, name in enumerate(comp_names):
                ax.plot(range(bs), r[metric][:, c], linewidth=1.5, color=colors[c], label=name, marker='o', markersize=2)
            key = 'ml_all' if metric == 'ml_pct' else 'mc_all'
            ax.set_xlabel('Position in sorted sequence', fontsize=11)
            ax.set_ylabel(ylab, fontsize=10)
            ax.set_title(f'{tt}\n(' + ', '.join([f'{comp_names[c]}={r[key][c]:.1f}%' for c in range(4)]) + ')',
                         fontsize=9, fontweight='bold')
            ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
        fig.suptitle(f'{cfg_name} — {ylab}\n{NUM_INDICATIVE_TRIALS} sequences', fontsize=12, fontweight='bold')
        fig.tight_layout()
        fig.savefig(f'{out_dir}/{fname}.png', dpi=200, bbox_inches='tight'); plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x_pos = np.arange(4); width = 0.35
    for i, (lt, tt) in enumerate([('LN0','Without LayerNorm'),('LN1','With LayerNorm')]):
        if lt not in results: continue
        ax = axes[i]; r = results[lt]
        bars1 = ax.bar(x_pos - width/2, r['ml_all'], width, label='Matches model logit', color='steelblue')
        bars2 = ax.bar(x_pos + width/2, r['mc_all'], width, label='Matches correct', color='coral')
        ax.set_xticks(x_pos); ax.set_xticklabels(comp_names, fontsize=9)
        ax.set_ylabel('Match rate (%)'); ax.set_title(tt, fontweight='bold')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.2, axis='y')
        for bar in bars1: ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5, f'{bar.get_height():.1f}%', ha='center', fontsize=7)
        for bar in bars2: ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5, f'{bar.get_height():.1f}%', ha='center', fontsize=7)
    fig.suptitle(f'{cfg_name} — Indicative match rates', fontsize=12, fontweight='bold')
    fig.tight_layout(); fig.savefig(f'{out_dir}/indicative_summary_bars.png', dpi=200, bbox_inches='tight'); plt.close()

def plot_length_gen(out_dir, cfg_name, lg_results):
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    for lt, color, label in [('LN0', '#1f77b4', 'Without LayerNorm'), ('LN1', '#ff7f0e', 'With LayerNorm')]:
        if lt not in lg_results: continue
        accs = lg_results[lt]
        ax.plot(EVAL_LENGTHS[:len(accs)], [a*100 for a in accs], linewidth=2, color=color, label=label, marker='o', markersize=4)
    ax.set_xlabel('Sequence length', fontsize=12); ax.set_ylabel('Token accuracy (%)', fontsize=12)
    ax.set_title(f'{cfg_name} — Length generalization', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
    if lg_results.get('bs'):
        ax.axvline(lg_results['bs'], color='gray', linestyle='--', alpha=0.5, label=f'train length={lg_results["bs"]}')
        ax.legend(fontsize=11)
    fig.tight_layout(); fig.savefig(f'{out_dir}/length_generalization_full_seq.png', dpi=200, bbox_inches='tight'); plt.close()

def parse_config(plot_dir):
    m = re.match(r'plots_(V\d+)_(B\d+)_(LR[\de-]+)_(MI\d+)_E(\d+)_H(\d+)_L(\d+)_ckpt(\d+)', plot_dir)
    if not m: return None
    vs_str, bs_str, lr_str, mi_str = m.group(1), m.group(2), m.group(3), m.group(4)
    ckpt = int(m.group(8))
    base = f'{vs_str}_{bs_str}_{lr_str}_{mi_str}'
    return {
        'base': base, 'plot_dir': plot_dir, 'ckpt': ckpt,
        'ln0_dir': f'{base}_LN0_E{m.group(5)}_H{m.group(6)}_L{m.group(7)}',
        'ln1_dir': f'{base}_LN1_E{m.group(5)}_H{m.group(6)}_L{m.group(7)}',
        'ln0_ckpt': f'{base}_LN0_E{m.group(5)}_H{m.group(6)}_L{m.group(7)}_itr{ckpt}.pt',
        'ln1_ckpt': f'{base}_LN1_E{m.group(5)}_H{m.group(6)}_L{m.group(7)}_itr{ckpt}.pt',
    }

if __name__ == '__main__':
    plot_dirs = sorted([d for d in os.listdir(OUTPUTS) if d.startswith('plots_')])
    total = len(plot_dirs)
    done = 0
    skipped = 0

    for pdir in plot_dirs:
        out_dir = os.path.join(OUTPUTS, pdir)
        already_done = (os.path.exists(os.path.join(out_dir, 'indicative_match_logit.png')) and
                        os.path.exists(os.path.join(out_dir, 'length_generalization_full_seq.png')))
        if already_done:
            done += 1
            continue

        cfg = parse_config(pdir)
        if cfg is None:
            skipped += 1; continue

        ln0_path = os.path.join(OUTPUTS, cfg['ln0_dir'], cfg['ln0_ckpt'])
        ln1_path = os.path.join(OUTPUTS, cfg['ln1_dir'], cfg['ln1_ckpt'])

        if not os.path.exists(ln0_path) or not os.path.exists(ln1_path):
            print(f'SKIP {pdir}: missing checkpoints')
            skipped += 1; continue

        done += 1
        t0 = time.time()
        print(f'[{done}/{total}] {pdir} ...', end=' ', flush=True)

        ind_results = {}
        lg_results = {}

        for ln, lt, path in [(0, 'LN0', ln0_path), (1, 'LN1', ln1_path)]:
            try:
                model, config = load_model(path)
            except Exception as e:
                print(f'ERROR loading {path}: {e}')
                continue

            ml_pct, mc_pct, ml_all, mc_all = run_indicative(model, config, NUM_INDICATIVE_TRIALS)
            ind_results[lt] = {'ml_pct': ml_pct, 'mc_pct': mc_pct, 'ml_all': ml_all, 'mc_all': mc_all}

            accs = run_length_gen(model, config)
            lg_results[lt] = accs
            lg_results['bs'] = config.block_size

            del model
            torch.cuda.empty_cache()

        if ind_results:
            plot_indicative(out_dir, cfg['base'], ind_results)
        if lg_results:
            plot_length_gen(out_dir, cfg['base'], lg_results)

        elapsed = time.time() - t0
        print(f'{elapsed:.1f}s')

    print(f'\nDone! Processed {done}, skipped {skipped}')
