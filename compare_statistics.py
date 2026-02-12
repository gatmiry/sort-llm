"""
Compare the ratio of samples in ICLogit+ICScore category to CLogit+ICScore category.

Categories:
- ICLogit+ICScore: Model prediction WRONG, Attention max score points to WRONG token
- CLogit+ICScore: Model prediction CORRECT, Attention max score points to WRONG token

Ratio = (# samples with wrong prediction & wrong attention) / (# samples with correct prediction & wrong attention)
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Create output folder
os.makedirs('plots_comparison', exist_ok=True)

# Load data from both models
data1 = np.load('plots/statistics_data.npz')  # Original model (dec28_tbyt)
data2 = np.load('plots_N128_K16_L2_H1_E32/statistics_data.npz')  # Grid checkpoint (N128_K16_L2_H1_E32)

# Use each file's own thresholds
thresholds1 = data1['thresholds']
thresholds2 = data2['thresholds']

# Get the counts (these are fractions of block_size, representing proportion of samples)
iclogit_icscore_1 = data1['ave_iclogit_icscore_perthreshold']  # Wrong prediction, wrong attention
clogit_icscore_1 = data1['ave_clogit_icscore_perthreshold']    # Correct prediction, wrong attention

iclogit_icscore_2 = data2['ave_iclogit_icscore_perthreshold']
clogit_icscore_2 = data2['ave_clogit_icscore_perthreshold']

# Calculate ratio: ICLogit+ICScore / CLogit+ICScore for each model
# Ratio = (wrong prediction & wrong attention) / (correct prediction & wrong attention)
epsilon = 1e-10
ratio1 = iclogit_icscore_1 / (clogit_icscore_1 + epsilon)
ratio2 = iclogit_icscore_2 / (clogit_icscore_2 + epsilon)

# Plot comparison (each model uses its own thresholds)
plt.figure(figsize=(10, 6))
plt.plot(thresholds1, ratio1, marker='o', linewidth=2, markersize=8, label='model without layer normalization', color='#1f77b4')
plt.plot(thresholds2, ratio2, marker='s', linewidth=2, markersize=8, label='model with layer normalization', color='#ff7f0e')
plt.xlabel('Threshold')
plt.ylabel('Ratio: (Wrong Pred & Wrong Attn) / (Correct Pred & Wrong Attn)')
plt.title('Ratio of ICLogit+ICScore to CLogit+ICScore Samples')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('plots_comparison/ratio_comparison.png', dpi=150, bbox_inches='tight')
print('Plot saved to plots_comparison/ratio_comparison.png')
plt.close()

# Print the values
print("\n" + "="*60)
print("Model 1 (model without layer normalization):")
print("="*60)
print(f"  Thresholds:              {thresholds1}")
print(f"  ICLogit+ICScore (wrong pred, wrong attn): {iclogit_icscore_1}")
print(f"  CLogit+ICScore (correct pred, wrong attn): {clogit_icscore_1}")
print(f"  Ratio (ICLogit_ICScore / CLogit_ICScore): {ratio1}")

print("\n" + "="*60)
print("Model 2 (model with layer normalization):")
print("="*60)
print(f"  Thresholds:              {thresholds2}")
print(f"  ICLogit+ICScore (wrong pred, wrong attn): {iclogit_icscore_2}")
print(f"  CLogit+ICScore (correct pred, wrong attn): {clogit_icscore_2}")
print(f"  Ratio (ICLogit_ICScore / CLogit_ICScore): {ratio2}")
