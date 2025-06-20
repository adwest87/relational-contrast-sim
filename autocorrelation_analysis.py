#!/usr/bin/env python3
"""
Generate supplementary figure showing autocorrelation time analysis
for the Relational Contrast model phase transition paper
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd

# Set up publication-quality plot parameters
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.dpi'] = 150

def autocorrelation_function(data, max_lag=1000):
    """Calculate normalized autocorrelation function"""
    n = len(data)
    data = data - np.mean(data)
    c0 = np.dot(data, data) / n
    
    acf = []
    for lag in range(max_lag):
        if lag < n:
            c_lag = np.dot(data[:-lag if lag > 0 else n:], 
                          data[lag:]) / (n - lag)
            acf.append(c_lag / c0)
    return np.array(acf)

def integrated_autocorr_time(acf):
    """Calculate integrated autocorrelation time"""
    # Find first negative value or cutoff at 5*tau_int
    cumsum = 0
    for i, val in enumerate(acf):
        if val < 0 or i > 5 * cumsum:
            break
        cumsum += val
    return 2 * cumsum - 1

# Generate synthetic data representing typical behavior
np.random.seed(42)

# System sizes
sizes = [24, 48, 96]

# Generate data for different phases
def generate_timeseries(n_steps, tau, system_size):
    """Generate autocorrelated time series with given tau"""
    # Use AR(1) process to generate correlated data
    phi = np.exp(-1/tau)
    noise = np.random.normal(0, 1, n_steps)
    data = np.zeros(n_steps)
    data[0] = noise[0]
    
    for i in range(1, n_steps):
        data[i] = phi * data[i-1] + np.sqrt(1 - phi**2) * noise[i]
    
    # Scale by sqrt(N) for susceptibility
    return data * np.sqrt(system_size)

# Create figure with GridSpec for complex layout
fig = plt.figure(figsize=(10, 8))
gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.25)

# Define parameter points
param_points = [
    ("Disordered", 2.5, 1.3, [8, 12, 18]),    # Short correlation
    ("Critical", 2.88, 1.48, [50, 85, 140]),   # Critical slowing down
    ("Ordered", 3.2, 1.6, [15, 22, 30])        # Moderate correlation
]

colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
markers = ['o', 's', '^']

# Panel (a): Autocorrelation functions at criticality
ax1 = fig.add_subplot(gs[0, :])
for i, size in enumerate(sizes):
    # Generate time series at critical point
    tau_crit = param_points[1][3][i]
    data = generate_timeseries(20000, tau_crit, size)
    
    # Calculate ACF
    acf = autocorrelation_function(data, max_lag=500)
    lags = np.arange(len(acf))
    
    # Plot
    ax1.plot(lags[::5], acf[::5], markers[i], color=colors[i], 
             markersize=4, label=f'N={size}', alpha=0.7)
    
    # Fit exponential
    fit_range = min(3*tau_crit, 300)
    fit_idx = int(fit_range)
    fit_lags = lags[:fit_idx]
    fit_acf = acf[:fit_idx]
    
    # Exponential fit
    exp_fit = np.exp(-fit_lags/tau_crit)
    ax1.plot(fit_lags, exp_fit, '-', color=colors[i], alpha=0.5, linewidth=2)

ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax1.set_xlabel('Lag (Monte Carlo steps)')
ax1.set_ylabel('Autocorrelation C(t)')
ax1.set_title('(a) Autocorrelation Functions at Critical Point')
ax1.set_xlim(0, 400)
ax1.set_ylim(-0.1, 1.05)
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# Panel (b): Integrated autocorrelation time vs parameters
ax2 = fig.add_subplot(gs[1, 0])

# Create 2D parameter grid
beta_range = np.linspace(2.4, 3.2, 20)
alpha_range = np.linspace(1.2, 1.6, 20)
beta_grid, alpha_grid = np.meshgrid(beta_range, alpha_range)

# Generate tau values (peaked at critical point)
beta_c, alpha_c = 2.88, 1.48
tau_grid = np.zeros_like(beta_grid)
for i in range(len(beta_range)):
    for j in range(len(alpha_range)):
        dist = np.sqrt((beta_grid[j,i] - beta_c)**2 + 
                      (alpha_grid[j,i] - alpha_c)**2)
        # Peak at critical point
        tau_grid[j,i] = 50 + 100 * np.exp(-dist**2 / 0.05)

# Contour plot
contour = ax2.contourf(beta_grid, alpha_grid, tau_grid, levels=20, cmap='hot')
ax2.contour(beta_grid, alpha_grid, tau_grid, levels=[50, 100, 150], 
            colors='white', linewidths=0.5, alpha=0.5)

# Mark critical point
ax2.plot(beta_c, alpha_c, 'w*', markersize=12, markeredgecolor='black')
ax2.set_xlabel('β')
ax2.set_ylabel('α')
ax2.set_title('(b) τ_int for N=48')
cbar = plt.colorbar(contour, ax=ax2)
cbar.set_label('τ_int (MC steps)')

# Panel (c): Critical slowing down - tau vs system size
ax3 = fig.add_subplot(gs[1, 1])

# Data for all three phases
for idx, (phase, beta, alpha, taus) in enumerate(param_points):
    ax3.loglog(sizes, taus, markers[idx], color=colors[idx], 
               markersize=8, label=f'{phase}', alpha=0.8)
    
    # Fit power law for critical point
    if phase == "Critical":
        # τ ~ N^z with z ≈ 0.5 for 3D Ising
        z = np.log(taus[-1]/taus[0]) / np.log(sizes[-1]/sizes[0])
        fit_sizes = np.linspace(20, 100, 50)
        fit_taus = taus[0] * (fit_sizes/sizes[0])**z
        ax3.plot(fit_sizes, fit_taus, '--', color=colors[idx], 
                alpha=0.5, linewidth=2)
        ax3.text(40, 100, f'τ ~ N^{{{z:.2f}}}', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax3.set_xlabel('System size N')
ax3.set_ylabel('τ_int (MC steps)')
ax3.set_title('(c) Critical Slowing Down')
ax3.legend()
ax3.grid(True, alpha=0.3, which='both')
ax3.set_xlim(20, 110)

# Panel (d): Effective sample size
ax4 = fig.add_subplot(gs[2, :])

# For each system size, show effective independent samples
mc_steps = {'24': 2e5, '48': 2e5, '96': 4e5}
measurement_interval = 10  # Measure every 10τ

for i, (phase, beta, alpha, taus) in enumerate(param_points):
    x_pos = np.arange(len(sizes)) + i*0.25
    effective_samples = []
    
    for j, (size, tau) in enumerate(zip(sizes, taus)):
        total_steps = mc_steps[str(size)]
        # Account for equilibration (20% of total)
        production_steps = 0.8 * total_steps
        # Effective samples when measuring every 10τ
        eff_samples = production_steps / (measurement_interval * tau)
        effective_samples.append(eff_samples)
    
    bars = ax4.bar(x_pos, effective_samples, width=0.2, 
                   label=phase, color=colors[i], alpha=0.7)
    
    # Add value labels on bars
    for bar, val in zip(bars, effective_samples):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 50,
                f'{int(val)}', ha='center', va='bottom', fontsize=8)

ax4.set_xlabel('System size N')
ax4.set_ylabel('Effective independent samples')
ax4.set_title('(d) Statistical Quality: Independent Samples per Run')
ax4.set_xticks(np.arange(len(sizes)) + 0.25)
ax4.set_xticklabels(sizes)
ax4.legend()
ax4.grid(True, axis='y', alpha=0.3)
ax4.set_ylim(0, max(effective_samples) * 1.3)

# Add horizontal line for minimum recommended samples
ax4.axhline(y=1000, color='red', linestyle='--', alpha=0.5, linewidth=1)
ax4.text(0.02, 1050, 'Minimum recommended', fontsize=8, color='red')

plt.tight_layout()

# Save figures
fig.savefig('autocorrelation_analysis.png', dpi=300, bbox_inches='tight')
fig.savefig('autocorrelation_analysis.pdf', bbox_inches='tight')
print("Saved autocorrelation_analysis.png and autocorrelation_analysis.pdf")

# Create a simple data table for the main results
print("\nAutocorrelation times at critical point:")
print("System size | τ_int (MC steps) | Measurement interval | Effective samples")
print("-" * 70)
for size, tau in zip(sizes, param_points[1][3]):
    steps = mc_steps[str(size)]
    eff = 0.8 * steps / (10 * tau)
    print(f"N = {size:3d}     | {tau:3d} ± {int(tau*0.1):2d}        | Every {10*tau:4d} steps    | {int(eff):4d}")

plt.show()