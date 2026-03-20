import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

block_size = 32
vocab_size = 128

data_without = np.load('plots_withoutlayernorm/statistics_data.npz')
avg_dist_without = data_without['ave_average_dist_perlocation']
random_prev_dist_without = data_without['ave_random_prev_dist_perlocation']

# Compute uniform random baseline per location
num_baseline_trials = 10000
uniform_dist = np.zeros(block_size)
for _ in range(num_baseline_trials):
    perm = np.random.permutation(vocab_size)[:block_size]
    sorted_perm = np.sort(perm)
    for j in range(block_size):
        v = sorted_perm[j]
        uniform_dist[j] += np.mean(np.abs(np.arange(vocab_size) - v))
uniform_dist /= num_baseline_trials

locs = np.arange(block_size)

fig, ax = plt.subplots(figsize=(7, 4))

ax.plot(locs, avg_dist_without, linewidth=2.2, marker='o', markersize=4,
        color='#6a3d9a', label='Candidate set avg distance')
ax.plot(locs, random_prev_dist_without, linewidth=2.2, marker='^', markersize=4,
        color='#2c7bb6', label='Random previous token distance')
ax.plot(locs, uniform_dist, linewidth=2.2, marker='s', markersize=4,
        color='#d7191c', label='Uniform random baseline')

ax.set_xlabel('Position in output sequence', fontsize=12)
ax.set_ylabel('Average distance to correct next number', fontsize=12)
ax.set_title('Candidate set distance vs baselines (Without LN)', fontsize=13, fontweight='bold')
ax.legend(fontsize=9, framealpha=0.9, loc='best')
ax.grid(True, alpha=0.2, linestyle=':')
ax.tick_params(labelsize=11)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout()

os.makedirs('plots_comparison', exist_ok=True)
output_path = 'plots_comparison/compare_distance_perlocation.png'
fig.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()
print(f'Plot saved to {output_path}')
