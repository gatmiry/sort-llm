from next_num_statistics import get_statistics, block_size
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

thresholds = [0.01, 0.05, 0.1, 0.15, 0.2]
threshold_index = 2
num_tries = 10

ave_average_dist_perthreshold = np.zeros(len(thresholds))
ave_max_dist_perthreshold = np.zeros(len(thresholds))
ave_average_dist_perlocation = np.zeros(block_size)
ave_max_dist_perlocation = np.zeros(block_size)

for counter in range(num_tries):
    _, _, _, _, am_perlocation, am_perthreshold = get_statistics(thresholds, threshold_index)
    average_dist_perthreshold, max_dist_perthreshold = am_perthreshold
    average_dist_perlocation, max_dist_perlocation = am_perlocation
    
    ave_average_dist_perthreshold += average_dist_perthreshold
    ave_max_dist_perthreshold += max_dist_perthreshold
    ave_average_dist_perlocation += average_dist_perlocation
    ave_max_dist_perlocation += max_dist_perlocation

ave_average_dist_perthreshold /= num_tries
ave_max_dist_perthreshold /= num_tries
ave_average_dist_perlocation /= num_tries
ave_max_dist_perlocation /= num_tries

# Plot average and max distance per threshold
plt.figure(figsize=(8, 6))
plt.plot(thresholds, ave_average_dist_perthreshold, marker='o', label='Average Distance', linewidth=2, markersize=8)
plt.plot(thresholds, ave_max_dist_perthreshold, marker='s', label='Max Distance', linewidth=2, markersize=8)
plt.xlabel('Threshold')
plt.ylabel('Distance')
plt.title('Distance per Threshold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('plots/distance_perthreshold.png', dpi=150, bbox_inches='tight')
print('Plot saved to plots/distance_perthreshold.png')
plt.close()

# Plot average and max distance per location
plt.figure(figsize=(8, 6))
location_indices = np.arange(block_size)
plt.plot(location_indices, ave_average_dist_perlocation, marker='o', label='Average Distance', linewidth=2, markersize=8)
plt.plot(location_indices, ave_max_dist_perlocation, marker='s', label='Max Distance', linewidth=2, markersize=8)
plt.xlabel('Location Index (within block)')
plt.ylabel('Distance')
plt.title('Distance per Location')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('plots/distance_perlocation.png', dpi=150, bbox_inches='tight')
print('Plot saved to plots/distance_perlocation.png')
plt.close()

