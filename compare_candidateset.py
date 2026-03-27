"""
Compare the average count per threshold (candidate set size) between models
with and without layer normalization.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# Create output folder
os.makedirs('plots_comparison', exist_ok=True)

# Load data from both folders
data1 = np.load('plots/statistics_data.npz')
data2 = np.load('plots_N128_K16_L2_H1_E32/statistics_data.npz')

thresholds1 = data1['thresholds']
thresholds2 = data2['thresholds']
count1 = data1['ave_count_perthreshold']
count2 = data2['ave_count_perthreshold']

# Plot comparison
fig, ax = plt.subplots(figsize=(4.5, 4))
ax.plot(thresholds1, count1, marker='o', linewidth=2.2, markersize=7, label='Without LayerNorm', color='#2c7bb6')
ax.plot(thresholds2, count2, marker='s', linewidth=2.2, markersize=7, label='With LayerNorm', color='#d7191c')
ax.set_xlabel('Threshold', fontsize=12)
ax.set_ylabel('Average size of candidate set', fontsize=12)
ax.set_title('Average size of candidate set per threshold', fontsize=13, fontweight='bold')
ax.legend(fontsize=10, framealpha=0.9)
ax.grid(True, alpha=0.2, linestyle=':')
ax.tick_params(labelsize=11)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout()
fig.savefig('plots_comparison/count_comparison.png', dpi=300, bbox_inches='tight')
print('Plot saved to plots_comparison/count_comparison.png')
plt.close()

# Print the values
print("\n" + "="*60)
print("Model without layer normalization:")
print("="*60)
print(f"  Thresholds: {thresholds1}")
print(f"  Average counts: {count1}")

print("\n" + "="*60)
print("Model with layer normalization:")
print("="*60)
print(f"  Thresholds: {thresholds2}")
print(f"  Average counts: {count2}")
