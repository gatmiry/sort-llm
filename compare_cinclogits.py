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

total_icscore_without = np.mean(clogit_icscore_without + iclogit_icscore_without)
total_icscore_with = np.mean(clogit_icscore_with + iclogit_icscore_with)

epsilon = 1e-10
correction_without = np.sum(clogit_icscore_without) / (np.sum(clogit_icscore_without + iclogit_icscore_without) + epsilon)
correction_with = np.sum(clogit_icscore_with) / (np.sum(clogit_icscore_with + iclogit_icscore_with) + epsilon)

os.makedirs('plots_comparison', exist_ok=True)

fig, ax = plt.subplots(figsize=(5, 4.2))

bar_width = 0.32
x = np.array([0, 1])

bars1 = ax.bar(x[0] - bar_width / 2, total_icscore_without, bar_width,
               color='#6a3d9a', label='Without LayerNorm')
bars2 = ax.bar(x[0] + bar_width / 2, total_icscore_with, bar_width,
               color='#e6850e', label='With LayerNorm')

bars3 = ax.bar(x[1] - bar_width / 2, correction_without, bar_width,
               color='#6a3d9a')
bars4 = ax.bar(x[1] + bar_width / 2, correction_with, bar_width,
               color='#e6850e')

for bar in [bars1, bars2, bars3, bars4]:
    for b in bar:
        height = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, height + 0.008,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(['Fraction of\nincorrect scores', 'Logit correction ratio\namong incorrect scores'],
                    fontsize=11)
ax.set_ylabel('Fraction', fontsize=12)
ax.set_title('Incorrect scores & logit correction ratio', fontsize=13, fontweight='bold')
ax.legend(fontsize=10, framealpha=0.9)
ax.grid(True, axis='y', alpha=0.2, linestyle=':')
ax.tick_params(labelsize=11)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylim(0, max(total_icscore_without, total_icscore_with, correction_without, correction_with) * 1.18)
fig.tight_layout()

output_path = 'plots_comparison/compare_cinclogits.png'
fig.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()
print(f'Plot saved to {output_path}')

print(f'\nFraction of ICScores — Without LN: {total_icscore_without:.4f}, With LN: {total_icscore_with:.4f}')
print(f'Correction ratio     — Without LN: {correction_without:.4f}, With LN: {correction_with:.4f}')
