#!/usr/bin/env python3
"""Check the quality of FSS data"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read all data
df24 = pd.read_csv('fss_data/results_n24.csv')
df48 = pd.read_csv('fss_data/results_n48.csv')
df96 = pd.read_csv('fss_data/results_n96.csv')

fig, axes = plt.subplots(3, 3, figsize=(15, 12))

# For each system size, show key diagnostics
for idx, (size, df) in enumerate([(24, df24), (48, df48), (96, df96)]):
    
    # 1. Acceptance rate distribution
    ax = axes[idx, 0]
    ax.hist(df['acceptance'], bins=30, alpha=0.7, edgecolor='black')
    ax.axvline(0.5, color='red', linestyle='--', label='Optimal')
    ax.set_xlabel('Acceptance Rate')
    ax.set_ylabel('Count')
    ax.set_title(f'N={size}: Acceptance Rates')
    ax.set_xlim(0, 1)
    mean_acc = df['acceptance'].mean()
    ax.text(0.05, 0.9, f'Mean: {mean_acc:.3f}', transform=ax.transAxes)
    
    # 2. Autocorrelation times
    ax = axes[idx, 1]
    ax.hist(df['autocorr_time'], bins=30, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Autocorrelation Time')
    ax.set_ylabel('Count')
    ax.set_title(f'N={size}: Autocorrelation Times')
    mean_tau = df['autocorr_time'].mean()
    ax.text(0.6, 0.9, f'Mean: {mean_tau:.1f}', transform=ax.transAxes)
    
    # 3. Susceptibility map
    ax = axes[idx, 2]
    # Create 2D histogram
    beta_bins = np.linspace(df['beta'].min(), df['beta'].max(), 12)
    alpha_bins = np.linspace(df['alpha'].min(), df['alpha'].max(), 12)
    
    # Pivot for heatmap
    pivot = df.pivot_table(values='susceptibility', index='beta', columns='alpha', aggfunc='mean')
    im = ax.imshow(pivot, origin='lower', aspect='auto', cmap='hot',
                   extent=[pivot.columns.min(), pivot.columns.max(),
                          pivot.index.min(), pivot.index.max()])
    ax.set_xlabel('α')
    ax.set_ylabel('β')
    ax.set_title(f'N={size}: Susceptibility Map')
    plt.colorbar(im, ax=ax)
    
    # Mark the peak
    peak_idx = df['susceptibility'].idxmax()
    peak = df.loc[peak_idx]
    ax.plot(peak['alpha'], peak['beta'], 'b*', markersize=15)

plt.suptitle('Data Quality Check', fontsize=16)
plt.tight_layout()
plt.savefig('data_quality_check.png', dpi=150, bbox_inches='tight')
plt.show()

# Check for anomalies in N=96 data
print("\nChecking N=96 data for anomalies:")
print("-" * 50)

# Find points with unusually low susceptibility
low_chi = df96[df96['susceptibility'] < 15]
print(f"\nPoints with χ < 15 in N=96 data: {len(low_chi)}")
if len(low_chi) > 0:
    print("\nSample of low-χ points:")
    print(low_chi[['beta', 'alpha', 'susceptibility', 'mean_cos', 'acceptance']].head(10))

# Compare equilibration
print("\n\nEquilibration comparison:")
print(f"N=24: {113137} equilibration steps")
print(f"N=48: {80000} equilibration steps")  
print(f"N=96: {56568} equilibration steps")
print("\nN=96 has FEWER equilibration steps despite being 4x larger!")

# Check if susceptibility correlates with acceptance or autocorrelation
print("\n\nCorrelations in N=96 data:")
print(f"χ vs acceptance rate: {df96['susceptibility'].corr(df96['acceptance']):.3f}")
print(f"χ vs autocorr time: {df96['susceptibility'].corr(df96['autocorr_time']):.3f}")

# Find the "true" peak by looking at the highest values across all sizes
print("\n\nFinding consistent high-susceptibility region:")
for beta in [2.88, 2.89, 2.90, 2.91, 2.92]:
    for alpha in [1.46, 1.48, 1.50]:
        chi_vals = []
        for size, df in [(24, df24), (48, df48), (96, df96)]:
            mask = (np.abs(df['beta'] - beta) < 0.01) & (np.abs(df['alpha'] - alpha) < 0.01)
            subset = df[mask]
            if len(subset) > 0:
                chi_vals.append(subset['susceptibility'].max())
            else:
                chi_vals.append(np.nan)
        
        if not any(np.isnan(chi_vals)):
            ratio1 = chi_vals[1] / chi_vals[0] if chi_vals[0] > 0 else 0
            ratio2 = chi_vals[2] / chi_vals[1] if chi_vals[1] > 0 else 0
            print(f"\n(β={beta:.2f}, α={alpha:.2f}):")
            print(f"  χ values: {chi_vals[0]:.1f}, {chi_vals[1]:.1f}, {chi_vals[2]:.1f}")
            print(f"  Ratios: {ratio1:.2f}, {ratio2:.2f}")
            
            # Estimate γ/ν from these ratios
            if ratio1 > 0:
                gamma_nu_est = np.log(ratio1) / np.log(2)
                print(f"  γ/ν estimate: {gamma_nu_est:.2f}")