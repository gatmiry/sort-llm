import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

data_without = np.load('plots_withoutlayernorm/topk_inclusion_data.npz')
data_with = np.load('plots_withlayernorm/topk_inclusion_data.npz')

k_values = data_without['k_values']
pct_without = data_without['inclusion_percentages']
pct_with = data_with['inclusion_percentages']

os.makedirs('plots', exist_ok=True)
output_path = 'plots/compare_topk_inclusion.png'

fig, ax = plt.subplots(figsize=(4.5, 4))
ax.plot(k_values, pct_without, marker='o', linewidth=2.2, markersize=7, label='Without LayerNorm', color='#6a3d9a')
ax.plot(k_values, pct_with, marker='s', linewidth=2.2, markersize=7, label='With LayerNorm', color='#e6850e')
ax.set_xticks(k_values)
ax.set_xlabel('k (Top-k attention over previous positions)', fontsize=12)
ax.set_ylabel('Inclusion percentage (%)', fontsize=12)
ax.set_title('Top-k Inclusion of Next Number', fontsize=13, fontweight='bold')
ax.legend(fontsize=10, framealpha=0.9)
ax.grid(True, alpha=0.2, linestyle=':')
ax.tick_params(labelsize=11)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout()
fig.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()

print(f'Plot saved to {output_path}')
