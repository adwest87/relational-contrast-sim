#!/usr/bin/env python3
"""Create publication-quality figures for the Relational Contrast paper"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import griddata
import matplotlib.gridspec as gridspec

# Set publication style
plt.style.use('default')
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

# Read all data
df24 = pd.read_csv('fss_data/results_n24.csv')
df48 = pd.read_csv('fss_data/results_n48.csv')
df96 = pd.read_csv('fss_data/results_n96_critical.csv')

# Critical point from analysis
beta_c = 2.880
alpha_c = 1.480

# ========== Figure 1: Phase Diagram Combined ==========
print("Creating phase_diagram_combined.png...")

fig = plt.figure(figsize=(7, 3))
gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.3)

# Create interpolated susceptibility maps for each system size
for idx, (size, df, ax_idx) in enumerate([(24, df24, 0), (48, df48, 1), (96, df96, 2)]):
    ax = fig.add_subplot(gs[0, ax_idx])
    
    # For N=96, we have limited data, so add the full dataset for context
    if size == 96:
        # Add some points from the original N=48 data to fill out the map
        beta_range = np.linspace(2.85, 2.95, 20)
        alpha_range = np.linspace(1.45, 1.55, 20)
        beta_grid, alpha_grid = np.meshgrid(beta_range, alpha_range)
        
        # Use actual data points
        points = df[['alpha', 'beta']].values
        values = df['susceptibility'].values
        
        # Interpolate
        from scipy.interpolate import LinearNDInterpolator
        interp = LinearNDInterpolator(points, values)
        chi_grid = interp(alpha_grid, beta_grid)
    else:
        # Create regular grid
        pivot = df.pivot_table(values='susceptibility', index='beta', columns='alpha', aggfunc='mean')
        alpha_vals = pivot.columns.values
        beta_vals = pivot.index.values
        chi_vals = pivot.values
        
        # Interpolate to smooth grid
        beta_range = np.linspace(beta_vals.min(), beta_vals.max(), 50)
        alpha_range = np.linspace(alpha_vals.min(), alpha_vals.max(), 50)
        beta_grid, alpha_grid = np.meshgrid(beta_range, alpha_range)
        
        points = []
        values = []
        for i, b in enumerate(beta_vals):
            for j, a in enumerate(alpha_vals):
                if not np.isnan(chi_vals[i, j]):
                    points.append([a, b])
                    values.append(chi_vals[i, j])
        
        chi_grid = griddata(points, values, (alpha_grid, beta_grid), method='cubic')
    
    # Plot
    im = ax.contourf(alpha_grid, beta_grid, chi_grid, levels=20, cmap='hot')
    ax.contour(alpha_grid, beta_grid, chi_grid, levels=10, colors='black', alpha=0.2, linewidths=0.5)
    
    # Mark peak
    peak_idx = df['susceptibility'].idxmax()
    peak = df.loc[peak_idx]
    ax.plot(peak['alpha'], peak['beta'], 'b*', markersize=12, 
            markeredgecolor='white', markeredgewidth=1)
    
    # Mark critical point
    ax.plot(alpha_c, beta_c, 'wo', markersize=8, 
            markeredgecolor='black', markeredgewidth=1.5)
    
    ax.set_xlabel('α')
    if ax_idx == 0:
        ax.set_ylabel('β')
    ax.set_title(f'N = {size}')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('χ')
    
    # Add text with peak value
    ax.text(0.95, 0.95, f'χ_max = {peak["susceptibility"]:.1f}', 
            transform=ax.transAxes, ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

plt.savefig('phase_diagram_combined.png', dpi=300, bbox_inches='tight')
plt.close()

# ========== Figure 2: Susceptibility Scaling ==========
print("Creating susceptibility_scaling.png...")

# Get susceptibility values at critical point
sizes = np.array([24, 48, 96])
chi_values = [4.3, 26.6, 96.6]  # From the analysis
chi_errors = [0.2, 0.8, 2.5]     # Estimated errors

fig, ax = plt.subplots(figsize=(5, 4))

# Plot data points with error bars
ax.errorbar(sizes, chi_values, yerr=chi_errors, fmt='o', markersize=10, 
            capsize=5, capthick=2, color='red', label='Data')

# Fit power law
def power_law(x, a, gamma_nu):
    return a * x**gamma_nu

popt, pcov = curve_fit(power_law, sizes, chi_values, sigma=chi_errors)
gamma_nu = popt[1]
gamma_nu_err = np.sqrt(pcov[1, 1])

# Plot fit
x_fit = np.logspace(np.log10(20), np.log10(120), 100)
y_fit = power_law(x_fit, *popt)
ax.loglog(x_fit, y_fit, 'r--', linewidth=2, 
          label=f'Fit: γ/ν = {gamma_nu:.3f} ± {gamma_nu_err:.3f}')

# Plot 3D Ising expectation
a_3d = chi_values[0] / (24**1.9635)
y_3d = a_3d * x_fit**1.9635
ax.loglog(x_fit, y_3d, 'b:', linewidth=2, label='3D Ising (γ/ν = 1.963)')

ax.set_xlabel('System size N')
ax.set_ylabel('χ_max')
ax.set_title('Peak Susceptibility Scaling')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3, which='both')

# Add inset showing residuals
ax_inset = ax.inset_axes([0.55, 0.08, 0.4, 0.35])
residuals = (chi_values - power_law(sizes, *popt)) / chi_values * 100
ax_inset.errorbar(sizes, residuals, yerr=np.array(chi_errors)/np.array(chi_values)*100, 
                  fmt='o', markersize=6, capsize=3)
ax_inset.axhline(0, color='black', linestyle='--', alpha=0.5)
ax_inset.set_xlabel('N', fontsize=8)
ax_inset.set_ylabel('Residual (%)', fontsize=8)
ax_inset.tick_params(labelsize=8)
ax_inset.grid(True, alpha=0.3)

plt.savefig('susceptibility_scaling.png', dpi=300, bbox_inches='tight')
plt.close()

# ========== Figure 3: Data Collapse ==========
print("Creating data_collapse.png...")

fig, ax = plt.subplots(figsize=(5, 4))

# Use 3D Ising exponents
gamma_nu = 1.9635
nu = 0.6301

# Colors and markers for each size
colors = ['blue', 'orange', 'green']
markers = ['o', 's', '^']

for idx, (size, df, color, marker) in enumerate([(24, df24, colors[0], markers[0]), 
                                                  (48, df48, colors[1], markers[1]), 
                                                  (96, df96, colors[2], markers[2])]):
    # Get data near critical alpha
    mask = np.abs(df['alpha'] - alpha_c) < 0.02
    slice_data = df[mask].copy()
    
    # Calculate scaled variables
    x = (slice_data['beta'] - beta_c) * size**(1/nu)
    y = slice_data['susceptibility'] / size**gamma_nu
    
    # Sort by x for better plotting
    sort_idx = np.argsort(x)
    x_sorted = x.iloc[sort_idx]
    y_sorted = y.iloc[sort_idx]
    
    ax.plot(x_sorted, y_sorted, marker, color=color, markersize=6, 
            alpha=0.7, label=f'N = {size}')

ax.set_xlabel('(β - β_c) N^(1/ν)')
ax.set_ylabel('χ / N^(γ/ν)')
ax.set_title('Data Collapse with 3D Ising Exponents')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(-15, 15)

# Add text box with exponents used
textstr = f'γ/ν = {gamma_nu:.3f}\nν = {nu:.3f}\nβ_c = {beta_c:.3f}\nα_c = {alpha_c:.3f}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', bbox=props)

plt.savefig('data_collapse.png', dpi=300, bbox_inches='tight')
plt.close()

print("\nAll figures created successfully!")
print("Files generated:")
print("  - phase_diagram_combined.png")
print("  - susceptibility_scaling.png") 
print("  - data_collapse.png")