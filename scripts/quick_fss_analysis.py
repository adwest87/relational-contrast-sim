#!/usr/bin/env python3
"""Quick FSS analysis with just N=24 and N=48"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read data
df24 = pd.read_csv('fss_data/results_n24.csv')
df48 = pd.read_csv('fss_data/results_n48.csv')

print("Quick FSS Analysis")
print("=" * 50)

# Find peaks
peak24 = df24.loc[df24['susceptibility'].idxmax()]
peak48 = df48.loc[df48['susceptibility'].idxmax()]

print(f"\nN=24: χ_max = {peak24['susceptibility']:.2f} at β={peak24['beta']:.2f}, α={peak24['alpha']:.2f}")
print(f"N=48: χ_max = {peak48['susceptibility']:.2f} at β={peak48['beta']:.2f}, α={peak48['alpha']:.2f}")

# Simple exponent estimate
chi_ratio = peak48['susceptibility'] / peak24['susceptibility']
size_ratio = 48 / 24
gamma_over_nu = np.log(chi_ratio) / np.log(size_ratio)

print(f"\nPreliminary exponent estimate:")
print(f"γ/ν ≈ {gamma_over_nu:.3f}")
print(f"\nCompare to:")
print(f"  3D Ising: γ/ν = 1.963")
print(f"  3D XY:    γ/ν = 1.973")
print(f"  4D Ising: γ/ν = 2.000")

# Check Binder cumulant
# Get Binder values at the peak from each system
beta_c = 2.90
alpha_c = 1.50

binder24 = df24[(np.abs(df24['beta'] - beta_c) < 0.02) & 
                 (np.abs(df24['alpha'] - alpha_c) < 0.02)]['binder'].mean()
binder48 = df48[(np.abs(df48['beta'] - beta_c) < 0.02) & 
                 (np.abs(df48['alpha'] - alpha_c) < 0.02)]['binder'].mean()

print(f"\nBinder cumulants at (β={beta_c}, α={alpha_c}):")
print(f"  N=24: U = {binder24:.3f}")
print(f"  N=48: U = {binder48:.3f}")
print(f"  Difference: {abs(binder48 - binder24):.3f}")

if abs(binder48 - binder24) < 0.1:
    print("  → Good agreement! Likely close to critical point")
else:
    print("  → Some finite-size drift, may need to adjust critical point")

# Plot comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Susceptibility peaks
ax1.scatter([24, 48], [peak24['susceptibility'], peak48['susceptibility']], 
            s=100, c='red', zorder=5)
ax1.plot([24, 48], [peak24['susceptibility'], peak48['susceptibility']], 
         'r--', alpha=0.5)
ax1.set_xlabel('System size N')
ax1.set_ylabel('χ_max')
ax1.set_title('Peak Susceptibility Scaling')
ax1.loglog()
ax1.grid(True, alpha=0.3)
ax1.text(30, 10, f'γ/ν ≈ {gamma_over_nu:.3f}', fontsize=12, 
         bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5))

# Binder at critical point
ax2.scatter([24, 48], [binder24, binder48], s=100, c='blue', zorder=5)
ax2.plot([24, 48], [binder24, binder48], 'b--', alpha=0.5)
ax2.set_xlabel('System size N')
ax2.set_ylabel('Binder Cumulant')
ax2.set_title(f'Binder at (β={beta_c}, α={alpha_c})')
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0, 0.7])

plt.tight_layout()
plt.savefig('quick_fss_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nPlot saved as quick_fss_analysis.png")