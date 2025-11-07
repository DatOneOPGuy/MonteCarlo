"""
Test script to verify triangular distribution implementation.

This script generates samples from triangular distributions and visualizes
them to confirm the mode (peak) is correctly positioned.
"""

import numpy as np
import matplotlib.pyplot as plt

# Test parameters (matching Apple project)
g1_params = (-0.15, 0.08, 0.34)   # min, mode, max
g2_params = (-0.15, 0.15, 0.41)

n_samples = 100000
seed = 42

np.random.seed(seed)

# Generate samples
g1 = np.random.triangular(g1_params[0], g1_params[1], g1_params[2], size=n_samples)
g2 = np.random.triangular(g2_params[0], g2_params[1], g2_params[2], size=n_samples)

# Print statistics
print("=" * 70)
print("Triangular Distribution Verification")
print("=" * 70)
print(f"\nYear 1 Growth (g1):")
print(f"  Parameters: min={g1_params[0]:.2f}, mode={g1_params[1]:.2f}, max={g1_params[2]:.2f}")
print(f"  Sample mean: {np.mean(g1):.4f}")
print(f"  Sample median: {np.median(g1):.4f}")
print(f"  Sample std: {np.std(g1):.4f}")
print(f"  Min in sample: {np.min(g1):.4f}")
print(f"  Max in sample: {np.max(g1):.4f}")
print(f"  Mode check: Values near mode ({g1_params[1]:.2f}) should be most frequent")

print(f"\nYear 2 Growth (g2):")
print(f"  Parameters: min={g2_params[0]:.2f}, mode={g2_params[1]:.2f}, max={g2_params[2]:.2f}")
print(f"  Sample mean: {np.mean(g2):.4f}")
print(f"  Sample median: {np.median(g2):.4f}")
print(f"  Sample std: {np.std(g2):.4f}")
print(f"  Min in sample: {np.min(g2):.4f}")
print(f"  Max in sample: {np.max(g2):.4f}")
print(f"  Mode check: Values near mode ({g2_params[1]:.2f}) should be most frequent")

# Create visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Year 1 distribution
ax1.hist(g1, bins=60, density=True, alpha=0.7, color='steelblue', edgecolor='black')
ax1.axvline(g1_params[0], color='red', linestyle='--', linewidth=2, label=f'Min: {g1_params[0]:.2f}')
ax1.axvline(g1_params[1], color='green', linestyle='--', linewidth=2, label=f'Mode: {g1_params[1]:.2f}')
ax1.axvline(g1_params[2], color='red', linestyle='--', linewidth=2, label=f'Max: {g1_params[2]:.2f}')
ax1.axvline(np.mean(g1), color='orange', linestyle=':', linewidth=2, label=f'Mean: {np.mean(g1):.4f}')
ax1.set_xlabel('Growth Rate', fontsize=12)
ax1.set_ylabel('Density', fontsize=12)
ax1.set_title('Year 1 Growth Rate Distribution (Triangular)', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Year 2 distribution
ax2.hist(g2, bins=60, density=True, alpha=0.7, color='steelblue', edgecolor='black')
ax2.axvline(g2_params[0], color='red', linestyle='--', linewidth=2, label=f'Min: {g2_params[0]:.2f}')
ax2.axvline(g2_params[1], color='green', linestyle='--', linewidth=2, label=f'Mode: {g2_params[1]:.2f}')
ax2.axvline(g2_params[2], color='red', linestyle='--', linewidth=2, label=f'Max: {g2_params[2]:.2f}')
ax2.axvline(np.mean(g2), color='orange', linestyle=':', linewidth=2, label=f'Mean: {np.mean(g2):.4f}')
ax2.set_xlabel('Growth Rate', fontsize=12)
ax2.set_ylabel('Density', fontsize=12)
ax2.set_title('Year 2 Growth Rate Distribution (Triangular)', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('triangular_distribution_test.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Visualization saved to: triangular_distribution_test.png")
print("=" * 70)

