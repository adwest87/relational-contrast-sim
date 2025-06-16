#!/usr/bin/env python3
"""Analyze the improved narrow scan results"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# Read the data
df = pd.read_csv('improved_narrow_results.csv')

# Create figure with subplots
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Phase Transition Analysis - Improved Z-Variable Results', fontsize=16)

# 1. Susceptibility heatmap
ax = axes[0, 0]
pivot_chi = df.pivot_table(values='susceptibility', index='beta', columns='alpha')
im1 = ax.imshow(pivot_chi, origin='lower', aspect='auto', cmap='hot',
                extent=[pivot_chi.columns.min(), pivot_chi.columns.max(),
                        pivot_chi.index.min(), pivot_chi.index.max()])
ax.set_title('Susceptibility χ')
ax.set_xlabel('α')
ax.set_ylabel('β')
plt.colorbar(im1, ax=ax)

# Find peak
max_chi_idx = df['susceptibility'].idxmax()
peak_chi = df.loc[max_chi_idx]
ax.plot(peak_chi['alpha'], peak_chi['beta'], 'b*', markersize=15)
ax.text(peak_chi['alpha'], peak_chi['beta'], 
        f' χ={peak_chi["susceptibility"]:.2f}', 
        color='blue', fontweight='bold')

# 2. Mean cos(θ) heatmap
ax = axes[0, 1]
pivot_cos = df.pivot_table(values='mean_cos', index='beta', columns='alpha')
im2 = ax.imshow(pivot_cos, origin='lower', aspect='auto', cmap='coolwarm',
                extent=[pivot_cos.columns.min(), pivot_cos.columns.max(),
                        pivot_cos.index.min(), pivot_cos.index.max()])
ax.set_title('Order Parameter ⟨cos θ⟩')
ax.set_xlabel('α')
ax.set_ylabel('β')
plt.colorbar(im2, ax=ax)

# 3. Action variance as proxy for specific heat
ax = axes[0, 2]
# Normalize by system size (48^2 * 47 / 6 = 17,296 links)
df['specific_heat'] = df['std_action']**2 / 17296
pivot_c = df.pivot_table(values='specific_heat', index='beta', columns='alpha')
im3 = ax.imshow(pivot_c, origin='lower', aspect='auto', cmap='plasma',
                extent=[pivot_c.columns.min(), pivot_c.columns.max(),
                        pivot_c.index.min(), pivot_c.index.max()])
ax.set_title('Specific Heat C (from action variance)')
ax.set_xlabel('α')
ax.set_ylabel('β')
plt.colorbar(im3, ax=ax)

# 4. Slice through susceptibility peak
ax = axes[1, 0]
# Get slice at peak alpha
peak_alpha = peak_chi['alpha']
slice_data = df[np.abs(df['alpha'] - peak_alpha) < 0.05].sort_values('beta')
ax.plot(slice_data['beta'], slice_data['susceptibility'], 'o-', label=f'α≈{peak_alpha:.2f}')
ax.set_xlabel('β')
ax.set_ylabel('Susceptibility χ')
ax.set_title('χ vs β at critical α')
ax.grid(True, alpha=0.3)
ax.legend()

# 5. Autocorrelation time
ax = axes[1, 1]
pivot_tau = df.pivot_table(values='autocorr_time', index='beta', columns='alpha')
im5 = ax.imshow(pivot_tau, origin='lower', aspect='auto', cmap='viridis',
                extent=[pivot_tau.columns.min(), pivot_tau.columns.max(),
                        pivot_tau.index.min(), pivot_tau.index.max()])
ax.set_title('Autocorrelation Time τ')
ax.set_xlabel('α')
ax.set_ylabel('β')
plt.colorbar(im5, ax=ax)

# 6. Phase diagram
ax = axes[1, 2]
# Use susceptibility to identify phases
chi_threshold = df['susceptibility'].mean() + df['susceptibility'].std()
high_chi = df[df['susceptibility'] > chi_threshold]

# Scatter plot colored by susceptibility
scatter = ax.scatter(df['alpha'], df['beta'], c=df['susceptibility'], 
                    s=50, cmap='hot', edgecolors='black', linewidth=0.5)
ax.set_xlabel('α')
ax.set_ylabel('β')
ax.set_title('Phase Diagram')
plt.colorbar(scatter, ax=ax, label='χ')

# Mark the critical region
if len(high_chi) > 0:
    from scipy.spatial import ConvexHull
    points = high_chi[['alpha', 'beta']].values
    if len(points) > 2:
        hull = ConvexHull(points)
        for simplex in hull.simplices:
            ax.plot(points[simplex, 0], points[simplex, 1], 'b-', alpha=0.5)

plt.tight_layout()
plt.savefig('phase_transition_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Print summary statistics
print("=== Phase Transition Analysis Summary ===\n")

print(f"Susceptibility peak:")
print(f"  β = {peak_chi['beta']:.3f}, α = {peak_chi['alpha']:.3f}")
print(f"  χ_max = {peak_chi['susceptibility']:.3f}\n")

# Find specific heat peak
max_c_idx = df['specific_heat'].idxmax()
peak_c = df.loc[max_c_idx]
print(f"Specific heat peak:")
print(f"  β = {peak_c['beta']:.3f}, α = {peak_c['alpha']:.3f}")
print(f"  C_max = {peak_c['specific_heat']:.3f}\n")

print(f"Action statistics:")
print(f"  Mean: {df['mean_action'].mean():.1f}")
print(f"  Std: {df['mean_action'].std():.1f}")
print(f"  Range: [{df['mean_action'].min():.1f}, {df['mean_action'].max():.1f}]\n")

print(f"Acceptance rate: {df['acceptance'].mean():.3f} ± {df['acceptance'].std():.3f}")
print(f"Autocorrelation time: {df['autocorr_time'].mean():.1f} ± {df['autocorr_time'].std():.1f}")

# Export clean data for further analysis
clean_df = df[['beta', 'alpha', 'mean_cos', 'susceptibility', 'specific_heat', 'autocorr_time']]
clean_df.to_csv('phase_transition_data.csv', index=False)
print("\nCleaned data exported to phase_transition_data.csv")