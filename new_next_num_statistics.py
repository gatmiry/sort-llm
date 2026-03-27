from model_tbyt_3 import GPT, GPTConfig
import torch
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

block_size = 32
vocab_size = 128
device = 'cuda'
wlnorm = 'with'

config = GPTConfig(block_size=block_size, vocab_size=vocab_size, without_pos=True)
model = GPT(config)
model_state_dict = torch.load(
    f'./saved_models/2026-03-14_19-03-50_vocab128/march14-{wlnorm}layernorm-block_size:32-batch_size:4096-n_embd:64_head:1_layers:2_vocab_size:128_itr:80000_checkpoint.pt',
    map_location=device
)['model']
model.load_state_dict(model_state_dict)
model.to(device=device)
model.eval()


def get_batch(batch_size=1):
    def cat_sorted_tensor(x):
        vals, _ = torch.sort(x)
        return torch.cat((x, torch.tensor([vocab_size]), vals), dim=0)
    x = torch.stack([cat_sorted_tensor(torch.randperm(vocab_size)[:block_size]) for _ in range(batch_size)])
    return x


def topk_next_number_inclusion(k_values, num_tries):
    """For each k in k_values, compute % of decode positions where
    idx[0, j+1] appears among the top-k attention targets (over previous
    positions k < j) of position j."""
    hits_per_k = np.zeros(len(k_values), dtype=np.float64)
    total_positions = block_size * num_tries

    for trial in range(num_tries):
        idx = get_batch(1).to(device=device)
        with torch.no_grad():
            model(idx)

        for j in range(block_size, 2 * block_size):
            next_num = idx[0, j + 1].item()
            prev_scores = []

            for k in range(0, j):
                score = model.transformer.h[0].c_attn.attn[j, k].item()
                prev_scores.append((score, k))

            prev_scores.sort(key=lambda x: x[0], reverse=True)

            for i, top_k in enumerate(k_values):
                topk_indices = [k_idx for _, k_idx in prev_scores[:top_k]]
                if any(idx[0, k_idx].item() == next_num for k_idx in topk_indices):
                    hits_per_k[i] += 1

    return 100.0 * hits_per_k / total_positions


if __name__ == '__main__':
    k_values = list(range(1, 11))
    num_tries = 100
    pct = topk_next_number_inclusion(k_values, num_tries)

    for k, p in zip(k_values, pct):
        print(f'k={k:2d}  inclusion={p:.2f}%')

    data_dir = f'plots_{wlnorm}layernorm'
    os.makedirs(data_dir, exist_ok=True)
    data_file = f'{data_dir}/topk_inclusion_data.npz'
    np.savez(data_file,
             k_values=np.array(k_values),
             inclusion_percentages=pct)
    print(f'Data saved to {data_file}')

    os.makedirs('plots', exist_ok=True)
    output_path = 'plots/topk_next_number_inclusion_percentage.png'

    plt.figure(figsize=(8, 5))
    plt.plot(k_values, pct, marker='o', linewidth=2, markersize=7, color='#1f77b4')
    plt.xticks(k_values)
    plt.xlabel('k (Top-k attention over previous positions)')
    plt.ylabel('Inclusion percentage (%)')
    plt.title('Top-k Inclusion of Next Number')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'\nPlot saved to {output_path}')
