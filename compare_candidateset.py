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
plt.figure(figsize=(10, 6))
plt.plot(thresholds1, count1, marker='o', linewidth=2, markersize=8, label='model without layer normalization', color='#1f77b4')
plt.plot(thresholds2, count2, marker='s', linewidth=2, markersize=8, label='model with layer normalization', color='#ff7f0e')
plt.xlabel('Threshold')
plt.ylabel('Average Count per Threshold')
plt.title('Average Count per Threshold Comparison')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('plots_comparison/count_comparison.png', dpi=150, bbox_inches='tight')
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
