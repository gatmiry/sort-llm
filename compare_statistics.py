"""
Compare ICLogit+ICScore to CLogit+ICScore ratio between two models.
Loads saved statistics from both next_num_statistics.py and next_num_statistics_loadcheckpoint.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Create output folder
os.makedirs('plots_comparison', exist_ok=True)

# Load data from both models
data1 = np.load('plots/statistics_data.npz')  # Original model (N128_K32)
data2 = np.load('plots_N256_K16_L2_H1_E32/statistics_data.npz')  # Load checkpoint model (N256_K16)

thresholds = data1['thresholds']

# Calculate ratio: ICLogit+ICScore / CLogit+ICScore for each model
# Avoid division by zero
epsilon = 1e-10
ratio1 = data1['ave_iclogit_icscore_perthreshold'] / (data1['ave_clogit_icscore_perthreshold'] + epsilon)
ratio2 = data2['ave_iclogit_icscore_perthreshold'] / (data2['ave_clogit_icscore_perthreshold'] + epsilon)

# Plot comparison
plt.figure(figsize=(10, 6))
plt.plot(thresholds, ratio1, marker='o', linewidth=2, markersize=8, label='N128_K32 (original)', color='#1f77b4')
plt.plot(thresholds, ratio2, marker='s', linewidth=2, markersize=8, label='N256_K16 (load_checkpoint)', color='#ff7f0e')
plt.xlabel('Threshold')
plt.ylabel('Ratio: ICLogit+ICScore / CLogit+ICScore')
plt.title('Comparison of ICLogit+ICScore to CLogit+ICScore Ratio')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('plots_comparison/ratio_comparison.png', dpi=150, bbox_inches='tight')
print('Plot saved to plots_comparison/ratio_comparison.png')
plt.close()

# Print the values
print("\nModel 1 (N128_K32):")
print(f"  Thresholds: {thresholds}")
print(f"  ICLogit+ICScore: {data1['ave_iclogit_icscore_perthreshold']}")
print(f"  CLogit+ICScore: {data1['ave_clogit_icscore_perthreshold']}")
print(f"  Ratio: {ratio1}")

print("\nModel 2 (N256_K16):")
print(f"  Thresholds: {thresholds}")
print(f"  ICLogit+ICScore: {data2['ave_iclogit_icscore_perthreshold']}")
print(f"  CLogit+ICScore: {data2['ave_clogit_icscore_perthreshold']}")
print(f"  Ratio: {ratio2}")
