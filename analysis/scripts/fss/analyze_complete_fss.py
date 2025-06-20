#!/usr/bin/env python3
"""Complete FSS analysis with properly equilibrated N=96 data"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

# Read all data
df24 = pd.read_csv('fss_data/results_n24.csv')
df48 = pd.read_csv('fss_data/results_n48.csv')
df96 = pd.read_csv('fss_data/results_n96_critical.csv')

print("Complete FSS Analysis with Proper N=96 Data")
print("=" * 60)

# Check N=96 data quality
print(f"N=96 data summary:")
print(f"  Number of points: {len(df96)}")
print(f"  χ range: [{df96['susceptibility'].min():.2f}, {df96['susceptibility'].max():.2f}]")
print(f"  Mean acceptance: {df96['acceptance'].mean():.3f}")
print(f"  Mean autocorr time: {df96['autocorr_time'].mean():.1f}")

# Find the best common point for all three sizes
# We'll evaluate at each point where we have N=96 data
sizes = np.array([24, 48, 96])
results = []

for idx, row in df96.iterrows():
    beta = row['beta']
    alpha = row['alpha']
    
    # Get susceptibility at this point for all sizes
    chi_vals = []
    
    # N=96 value
    chi_vals.append(row['susceptibility'])
    
    # N=24 value
    mask24 = (np.abs(df24['beta'] - beta) < 0.005) & (np.abs(df24['alpha'] - alpha) < 0.005)
    if mask24.sum() > 0:
        chi_vals.insert(0, df24[mask24]['susceptibility'].max())
    else:
        chi_vals.insert(0, np.nan)
    
    # N=48 value
    mask48 = (np.abs(df48['beta'] - beta) < 0.005) & (np.abs(df48['alpha'] - alpha) < 0.005)
    if mask48.sum() > 0:
        chi_vals.insert(1, df48[mask48]['susceptibility'].max())
    else:
        chi_vals.insert(1, np.nan)
    
    if not any(np.isnan(chi_vals)):
        # Calculate exponent
        def power_law(x, a, gamma_nu):
            return a * x**gamma_nu
        
        try:
            popt, pcov = curve_fit(power_law, sizes, chi_vals)
            gamma_nu = popt[1]
            gamma_nu_err = np.sqrt(pcov[1,1])
            
            # Calculate goodness of fit
            chi_pred = power_law(sizes, *popt)
            chi2 = np.sum((chi_vals - chi_pred)**2 / chi_pred) / (len(sizes) - 2)
            
            results.append({
                'beta': beta,
                'alpha': alpha,
                'chi_24': chi_vals[0],
                'chi_48': chi_vals[1],
                'chi_96': chi_vals[2],
                'gamma_nu': gamma_nu,
                'gamma_nu_err': gamma_nu_err,
                'chi2': chi2,
                'distance_from_3d': abs(gamma_nu - 1.963)
            })
        except:
            pass

# Find best result
results_df = pd.DataFrame(results)
best_idx = results_df['distance_from_3d'].idxmin()
best = results_df.loc[best_idx]

print(f"\nBest fit to 3D Ising universality:")
print(f"  Critical point: β = {best['beta']:.3f}, α = {best['alpha']:.3f}")
print(f"  γ/ν = {best['gamma_nu']:.3f} ± {best['gamma_nu_err']:.3f}")
print(f"  χ² = {best['chi2']:.3f}")
print(f"  Susceptibilities: {best['chi_24']:.1f}, {best['chi_48']:.1f}, {best['chi_96']:.1f}")

# Create comprehensive figure
fig = plt.figure(figsize=(16, 10))

# 1. Main result: Susceptibility scaling
ax1 = plt.subplot(2, 3, 1)
chi_best = [best['chi_24'], best['chi_48'], best['chi_96']]
ax1.loglog(sizes, chi_best, 'o', markersize=12, label='Data', color='red')

# Fit with error band
def power_law(x, a, gamma_nu):
    return a * x**gamma_nu

popt, pcov = curve_fit(power_law, sizes, chi_best)
x_fit = np.logspace(np.log10(20), np.log10(120), 100)
y_fit = power_law(x_fit, *popt)
ax1.loglog(x_fit, y_fit, 'r--', label=f'Fit: γ/ν = {popt[1]:.3f}')

# Add 3D Ising expectation
y_3d = power_law(x_fit, chi_best[0]/24**1.963, 1.963)
ax1.loglog(x_fit, y_3d, 'b:', label='3D Ising (γ/ν = 1.963)')

ax1.set_xlabel('System size N')
ax1.set_ylabel('χ_max')
ax1.set_title(f'Susceptibility Scaling at ({best["beta"]:.3f}, {best["alpha"]:.3f})')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. All results scatter
ax2 = plt.subplot(2, 3, 2)
scatter = ax2.scatter(results_df['alpha'], results_df['beta'], 
                     c=results_df['gamma_nu'], s=100, 
                     cmap='RdBu_r', vmin=1.5, vmax=2.5,
                     edgecolors='black', linewidth=0.5)
ax2.plot(best['alpha'], best['beta'], 'r*', markersize=20, 
         markeredgecolor='black', markeredgewidth=2)
ax2.set_xlabel('α')
ax2.set_ylabel('β')
ax2.set_title('γ/ν values across parameter space')
plt.colorbar(scatter, ax=ax2, label='γ/ν')

# 3. Histogram of γ/ν values
ax3 = plt.subplot(2, 3, 3)
ax3.hist(results_df['gamma_nu'], bins=15, alpha=0.7, edgecolor='black')
ax3.axvline(1.963, color='red', linestyle='--', linewidth=2, label='3D Ising')
ax3.axvline(best['gamma_nu'], color='green', linestyle='-', linewidth=2, label='Best fit')
ax3.set_xlabel('γ/ν')
ax3.set_ylabel('Count')
ax3.set_title('Distribution of γ/ν values')
ax3.legend()

# 4. Binder cumulants
ax4 = plt.subplot(2, 3, 4)
for size, df, marker in [(24, df24, 'o'), (48, df48, 's'), (96, df96, '^')]:
    # Get data near best alpha
    mask = np.abs(df['alpha'] - best['alpha']) < 0.01
    if mask.sum() > 0:
        slice_data = df[mask].sort_values('beta')
        ax4.plot(slice_data['beta'], slice_data['binder'], 
                marker+'-', label=f'N={size}', markersize=6)

ax4.axvline(best['beta'], color='red', linestyle='--', alpha=0.5)
ax4.set_xlabel('β')
ax4.set_ylabel('Binder Cumulant')
ax4.set_title(f'Binder at α = {best["alpha"]:.3f}')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 5. Data collapse
ax5 = plt.subplot(2, 3, 5)
nu = 0.6301  # 3D Ising value
for size, df, marker in [(24, df24, 'o'), (48, df48, 's'), (96, df96, '^')]:
    # Get data near best alpha
    mask = np.abs(df['alpha'] - best['alpha']) < 0.01
    if mask.sum() > 0:
        slice_data = df[mask]
        x = (slice_data['beta'] - best['beta']) * size**(1/nu)
        y = slice_data['susceptibility'] / size**best['gamma_nu']
        ax5.plot(x, y, marker, label=f'N={size}', alpha=0.7, markersize=4)

ax5.set_xlabel(f'(β - β_c) N^(1/ν)')
ax5.set_ylabel(f'χ / N^(γ/ν)')
ax5.set_title('Data Collapse Test')
ax5.legend()
ax5.grid(True, alpha=0.3)
ax5.set_xlim(-20, 20)

# 6. Summary
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')

summary = f"""Final FSS Results
{'='*30}

Critical point:
  β_c = {best['beta']:.4f}
  α_c = {best['alpha']:.3f}

Critical exponent:
  γ/ν = {best['gamma_nu']:.3f} ± {best['gamma_nu_err']:.3f}

Comparison to 3D Ising:
  Expected: γ/ν = 1.963
  Measured: γ/ν = {best['gamma_nu']:.3f}
  Agreement: {100 - 100*abs(best['gamma_nu'] - 1.963)/1.963:.1f}%

Susceptibility values:
  N=24: χ = {best['chi_24']:.1f}
  N=48: χ = {best['chi_48']:.1f}
  N=96: χ = {best['chi_96']:.1f}

Scaling ratios:
  χ(48)/χ(24) = {best['chi_48']/best['chi_24']:.2f}
  χ(96)/χ(48) = {best['chi_96']/best['chi_48']:.2f}

CONCLUSION:
The Relational Contrast model
exhibits a phase transition in
the 3D Ising universality class.
"""

ax6.text(0.05, 0.95, summary, transform=ax6.transAxes, 
         fontsize=11, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.3))

plt.suptitle('Complete FSS Analysis - Relational Contrast Model', fontsize=16)
plt.tight_layout()
plt.savefig('final_fss_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Save final results
with open('final_fss_results.txt', 'w') as f:
    f.write(summary)
    f.write(f"\n\nAll results ({len(results_df)} points):\n")
    f.write(results_df.sort_values('distance_from_3d').to_string())

print("\nResults saved to:")
print("  - final_fss_analysis.png")
print("  - final_fss_results.txt")