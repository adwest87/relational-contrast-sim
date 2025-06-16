#!/usr/bin/env python3
"""Visualize finite-size scaling results"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import glob

# Read all FSS data files
files = glob.glob('fss_data/results_n*.csv')
if not files:
    print("No FSS results found. Run the FSS scans first.")
    exit(1)

# Load data
all_data = []
for f in files:
    df = pd.read_csv(f)
    all_data.append(df)

# Combine and sort by system size
df_all = pd.concat(all_data)
sizes = sorted(df_all['n_nodes'].unique())

print(f"Found data for system sizes: {sizes}")

# Critical point
beta_c = 2.90
alpha_c = 1.50

# Create figure
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Finite-Size Scaling Analysis', fontsize=16)

# 1. Susceptibility scaling
ax = axes[0, 0]
for size in sizes:
    df_size = df_all[df_all['n_nodes'] == size]
    # Get data near critical point
    near_crit = df_size[(np.abs(df_size['beta'] - beta_c) < 0.05) & 
                        (np.abs(df_size['alpha'] - alpha_c) < 0.05)]
    if len(near_crit) > 0:
        ax.scatter([size], [near_crit['susceptibility'].max()], 
                  s=100, label=f'N={size}')

ax.set_xlabel('System size N')
ax.set_ylabel('χ_max')
ax.set_title('Peak Susceptibility Scaling')
ax.legend()
ax.loglog()
ax.grid(True, alpha=0.3)

# Fit χ ~ N^(γ/ν)
if len(sizes) >= 3:
    chi_max = []
    for size in sizes:
        df_size = df_all[df_all['n_nodes'] == size]
        chi_max.append(df_size['susceptibility'].max())
    
    def power_law(x, a, b):
        return a * x**b
    
    popt, _ = curve_fit(power_law, sizes, chi_max)
    x_fit = np.logspace(np.log10(min(sizes)), np.log10(max(sizes)), 50)
    ax.plot(x_fit, power_law(x_fit, *popt), 'r--', 
            label=f'γ/ν = {popt[1]:.3f}')
    ax.legend()

# 2. Binder cumulant crossing
ax = axes[0, 1]
for size in sizes:
    df_size = df_all[df_all['n_nodes'] == size]
    # Slice at α = α_c
    slice_data = df_size[np.abs(df_size['alpha'] - alpha_c) < 0.01].sort_values('beta')
    if len(slice_data) > 0:
        ax.plot(slice_data['beta'], slice_data['binder'], 
                'o-', label=f'N={size}')

ax.set_xlabel('β')
ax.set_ylabel('Binder Cumulant')
ax.set_title(f'Binder Cumulant at α={alpha_c:.2f}')
ax.legend()
ax.grid(True, alpha=0.3)

# 3. Mean cos θ (order parameter)
ax = axes[0, 2]
for size in sizes:
    df_size = df_all[df_all['n_nodes'] == size]
    # At critical point
    crit = df_size[(np.abs(df_size['beta'] - beta_c) < 0.01) & 
                   (np.abs(df_size['alpha'] - alpha_c) < 0.01)]
    if len(crit) > 0:
        ax.scatter([size], [crit['mean_cos'].mean()], 
                  s=100, label=f'N={size}')

ax.set_xlabel('System size N')
ax.set_ylabel('⟨cos θ⟩')
ax.set_title('Order Parameter at Critical Point')
ax.semilogx()
ax.grid(True, alpha=0.3)

# 4. Data collapse attempt for χ
ax = axes[1, 0]
# Try to collapse χ/N^(γ/ν) vs (β-β_c)N^(1/ν)
# Assume γ/ν ≈ 1.96 (3D Ising) and 1/ν ≈ 1.59
gamma_over_nu = 1.96
one_over_nu = 1.59

for size in sizes:
    df_size = df_all[df_all['n_nodes'] == size]
    slice_data = df_size[np.abs(df_size['alpha'] - alpha_c) < 0.01]
    
    x = (slice_data['beta'] - beta_c) * size**one_over_nu
    y = slice_data['susceptibility'] / size**gamma_over_nu
    
    ax.plot(x, y, 'o', label=f'N={size}', alpha=0.7)

ax.set_xlabel('(β - β_c) N^(1/ν)')
ax.set_ylabel('χ / N^(γ/ν)')
ax.set_title('Susceptibility Data Collapse')
ax.legend()
ax.grid(True, alpha=0.3)

# 5. Autocorrelation time
ax = axes[1, 1]
for size in sizes:
    df_size = df_all[df_all['n_nodes'] == size]
    crit = df_size[(np.abs(df_size['beta'] - beta_c) < 0.01) & 
                   (np.abs(df_size['alpha'] - alpha_c) < 0.01)]
    if len(crit) > 0:
        ax.scatter([size], [crit['autocorr_time'].mean()], 
                  s=100, label=f'N={size}')

ax.set_xlabel('System size N')
ax.set_ylabel('τ')
ax.set_title('Autocorrelation Time')
ax.loglog()
ax.legend()
ax.grid(True, alpha=0.3)

# 6. Summary text
ax = axes[1, 2]
ax.axis('off')

summary = f"""FSS Analysis Summary
─────────────────────
Critical point:
  β_c = {beta_c:.3f}
  α_c = {alpha_c:.3f}

System sizes: {sizes}

Preliminary exponents:
  γ/ν ≈ {popt[1]:.3f} (from χ scaling)
  
Compare to:
  3D Ising: γ/ν = 1.963
  4D Ising: γ/ν = 2.000
  3D XY:    γ/ν = 1.973
"""

ax.text(0.1, 0.9, summary, transform=ax.transAxes, 
        fontsize=12, verticalalignment='top', fontfamily='monospace')

plt.tight_layout()
plt.savefig('fss_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Save critical exponents
with open('critical_exponents.txt', 'w') as f:
    f.write(summary)