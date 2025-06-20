#!/usr/bin/env python3
"""Debug the FSS analysis to understand the anomalous results"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read all data
df24 = pd.read_csv('fss_data/results_n24.csv')
df48 = pd.read_csv('fss_data/results_n48.csv')
df96 = pd.read_csv('fss_data/results_n96.csv')

print("Debugging FSS Analysis")
print("=" * 60)

# Check data integrity
print("\nData summary:")
for name, df in [("N=24", df24), ("N=48", df48), ("N=96", df96)]:
    print(f"\n{name}:")
    print(f"  Number of points: {len(df)}")
    print(f"  χ range: [{df['susceptibility'].min():.2f}, {df['susceptibility'].max():.2f}]")
    print(f"  β range: [{df['beta'].min():.2f}, {df['beta'].max():.2f}]")
    print(f"  α range: [{df['alpha'].min():.2f}, {df['alpha'].max():.2f}]")

# Find true peaks
print("\nPeak susceptibilities:")
peaks = {}
for size, df in [(24, df24), (48, df48), (96, df96)]:
    peak_idx = df['susceptibility'].idxmax()
    peak = df.loc[peak_idx]
    peaks[size] = peak
    print(f"N={size}: χ_max = {peak['susceptibility']:.2f} at β={peak['beta']:.3f}, α={peak['alpha']:.3f}")

# Check susceptibility scaling
sizes = np.array([24, 48, 96])
chi_max = np.array([peaks[s]['susceptibility'] for s in sizes])

print(f"\nSusceptibility values: {chi_max}")
print(f"Ratios: χ(48)/χ(24) = {chi_max[1]/chi_max[0]:.3f}, χ(96)/χ(48) = {chi_max[2]/chi_max[1]:.3f}")

# Manual calculation of γ/ν
# From 24 to 48
ratio1 = chi_max[1] / chi_max[0]
size_ratio1 = 48.0 / 24.0
gamma_nu_1 = np.log(ratio1) / np.log(size_ratio1)

# From 48 to 96
ratio2 = chi_max[2] / chi_max[1]
size_ratio2 = 96.0 / 48.0
gamma_nu_2 = np.log(ratio2) / np.log(size_ratio2)

# From 24 to 96
ratio3 = chi_max[2] / chi_max[0]
size_ratio3 = 96.0 / 24.0
gamma_nu_3 = np.log(ratio3) / np.log(size_ratio3)

print(f"\nManual γ/ν calculations:")
print(f"  From N=24→48: γ/ν = {gamma_nu_1:.3f}")
print(f"  From N=48→96: γ/ν = {gamma_nu_2:.3f}")
print(f"  From N=24→96: γ/ν = {gamma_nu_3:.3f}")
print(f"  Average: γ/ν = {np.mean([gamma_nu_1, gamma_nu_2, gamma_nu_3]):.3f}")

# Plot to visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Log-log plot
ax1.loglog(sizes, chi_max, 'o', markersize=12, label='Data')
ax1.set_xlabel('System size N')
ax1.set_ylabel('χ_max')
ax1.set_title('Susceptibility Scaling (log-log)')
ax1.grid(True, alpha=0.3)

# Add power law fits between pairs
for i in range(len(sizes)-1):
    x = np.array([sizes[i], sizes[i+1]])
    y = np.array([chi_max[i], chi_max[i+1]])
    ax1.loglog(x, y, '--', alpha=0.5)

# Check if there's a system size effect
ax2.plot(sizes, chi_max / sizes**1.96, 'o-', markersize=10)
ax2.set_xlabel('System size N')
ax2.set_ylabel('χ_max / N^1.96')
ax2.set_title('Scaled susceptibility (should be flat for γ/ν=1.96)')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('debug_fss_scaling.png', dpi=150)
plt.show()

# Check for data issues near critical point
print("\nChecking data near estimated critical point (β≈2.88, α≈1.50):")
for size, df in [(24, df24), (48, df48), (96, df96)]:
    mask = (np.abs(df['beta'] - 2.88) < 0.05) & (np.abs(df['alpha'] - 1.50) < 0.05)
    subset = df[mask]
    print(f"\nN={size}: {len(subset)} points near critical region")
    if len(subset) > 0:
        print(f"  χ range in critical region: [{subset['susceptibility'].min():.2f}, {subset['susceptibility'].max():.2f}]")