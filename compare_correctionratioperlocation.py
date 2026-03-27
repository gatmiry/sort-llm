import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

data_without = np.load('plots_withoutlayernorm/statistics_data.npz')
data_with = np.load('plots_withlayernorm/statistics_data.npz')

clogit_icscore_without = data_without['ave_clogit_icscore_perlocation']
iclogit_icscore_without = data_without['ave_iclogit_icscore_perlocation']

clogit_icscore_with = data_with['ave_clogit_icscore_perlocation']
iclogit_icscore_with = data_with['ave_iclogit_icscore_perlocation']

total_without = clogit_icscore_without + iclogit_icscore_without
total_with = clogit_icscore_with + iclogit_icscore_with

epsilon = 1e-10
correction_ratio_without = clogit_icscore_without / (total_without + epsilon)
correction_ratio_with = clogit_icscore_with / (total_with + epsilon)

block_size = len(correction_ratio_without)
locations = np.arange(block_size)

os.makedirs('plots_comparison', exist_ok=True)

fig, ax = plt.subplots(figsize=(4.5, 4))
ax.plot(locations, correction_ratio_without, linewidth=2.2, marker='o', markersize=5,
        label='Without LayerNorm', color='#6a3d9a')
ax.plot(locations, correction_ratio_with, linewidth=2.2, marker='s', markersize=5,
        label='With LayerNorm', color='#e6850e')
ax.set_xlabel('Position in output sequence', fontsize=12)
ax.set_ylabel('Logit correction ratio', fontsize=12)
ax.set_title('Fraction of ICScores with correct logits', fontsize=13, fontweight='bold')
ax.legend(fontsize=10, framealpha=0.9)
ax.grid(True, alpha=0.2, linestyle=':')
ax.tick_params(labelsize=11)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout()

output_path = 'plots_comparison/correction_ratio_perlocation.png'
fig.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()
print(f'Plot saved to {output_path}')

print('\nWithout LayerNorm - correction ratio per location:')
print(correction_ratio_without)
print('\nWith LayerNorm - correction ratio per location:')
print(correction_ratio_with)
