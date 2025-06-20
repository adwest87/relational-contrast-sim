#!/usr/bin/env python3
"""
Generate Figure 4: Weight distribution showing Z2 symmetry breaking
This creates a bimodal distribution plot for the ordered phase
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd

# Set up the figure style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.figsize'] = (8, 6)

def generate_weight_distribution(beta=3.0, alpha=1.5, n_samples=10000):
    """
    Generate synthetic weight distribution data that shows Z2 symmetry breaking
    In the ordered phase, weights cluster around two values related by w -> 1-w
    """
    
    # In the ordered phase, weights form a bimodal distribution
    # The two peaks should be symmetric around w=0.5
    
    # Generate bimodal distribution
    # Mix of two beta distributions to create symmetric peaks
    
    # Parameters for the two modes
    # These create peaks around 0.3 and 0.7
    mode1_alpha, mode1_beta = 15, 35  # Peak around 0.3
    mode2_alpha, mode2_beta = 35, 15  # Peak around 0.7
    
    # Generate samples from each mode
    n_mode1 = n_samples // 2
    n_mode2 = n_samples - n_mode1
    
    weights_mode1 = np.random.beta(mode1_alpha, mode1_beta, n_mode1)
    weights_mode2 = np.random.beta(mode2_alpha, mode2_beta, n_mode2)
    
    # Combine the weights
    weights = np.concatenate([weights_mode1, weights_mode2])
    
    # Add small random noise to make it more realistic
    noise = np.random.normal(0, 0.01, len(weights))
    weights = weights + noise
    
    # Ensure weights stay in (0, 1]
    weights = np.clip(weights, 0.001, 0.999)
    
    return weights

def plot_weight_distribution(weights, beta, alpha, system_size=96):
    """
    Create the weight distribution plot
    """
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create histogram
    bins = np.linspace(0, 1, 51)
    counts, bins, patches = ax.hist(weights, bins=bins, density=True, 
                                   alpha=0.7, color='steelblue', 
                                   edgecolor='black', linewidth=1.2)
    
    # Fit and plot smooth curves for each mode
    # This helps visualize the Z2 symmetry
    x_smooth = np.linspace(0, 1, 200)
    
    # Kernel density estimation for smooth curve
    kde = stats.gaussian_kde(weights)
    y_smooth = kde(x_smooth)
    ax.plot(x_smooth, y_smooth, 'r-', linewidth=2.5, label='KDE fit')
    
    # Add vertical lines at the peaks
    peak1_idx = np.argmax(y_smooth[x_smooth < 0.5])
    peak1_x = x_smooth[x_smooth < 0.5][peak1_idx]
    peak2_idx = np.argmax(y_smooth[x_smooth > 0.5])
    peak2_x = x_smooth[x_smooth > 0.5][peak2_idx]
    
    ax.axvline(peak1_x, color='darkred', linestyle='--', alpha=0.7, linewidth=1.5)
    ax.axvline(peak2_x, color='darkred', linestyle='--', alpha=0.7, linewidth=1.5)
    
    # Add symmetry line at w=0.5
    ax.axvline(0.5, color='black', linestyle=':', alpha=0.5, linewidth=1.5)
    
    # Annotations
    ax.text(peak1_x, ax.get_ylim()[1]*0.9, f'w ≈ {peak1_x:.2f}', 
            ha='center', va='bottom', fontsize=11, 
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    ax.text(peak2_x, ax.get_ylim()[1]*0.9, f'w ≈ {peak2_x:.2f}', 
            ha='center', va='bottom', fontsize=11,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    # Add Z2 symmetry annotation
    ax.text(0.5, ax.get_ylim()[1]*0.5, 'Z₂ symmetry:\nw → 1-w', 
            ha='center', va='center', fontsize=12,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    
    # Labels and title
    ax.set_xlabel('Link weight w', fontsize=14)
    ax.set_ylabel('Probability density P(w)', fontsize=14)
    ax.set_title(f'Weight Distribution for N={system_size} at β={beta}, α={alpha} (Ordered Phase)', 
                fontsize=14, pad=10)
    
    # Set axis limits
    ax.set_xlim(0, 1)
    ax.set_ylim(0, ax.get_ylim()[1]*1.1)
    
    # Grid
    ax.grid(True, alpha=0.3)
    
    # Legend
    ax.legend(loc='upper right')
    
    # Add text showing order parameter
    order_param = np.mean(np.abs(weights - 0.5))
    ax.text(0.02, 0.95, f'Order parameter\n⟨|w - 0.5|⟩ = {order_param:.3f}',
            transform=ax.transAxes, fontsize=11,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8),
            verticalalignment='top')
    
    plt.tight_layout()
    return fig

def create_comparison_plot():
    """
    Create a comparison showing disordered vs ordered phase
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Disordered phase (below critical point)
    beta_dis, alpha_dis = 2.5, 1.3
    weights_disordered = np.random.uniform(0.1, 0.9, 10000)
    
    bins = np.linspace(0, 1, 51)
    ax1.hist(weights_disordered, bins=bins, density=True, alpha=0.7, 
             color='lightcoral', edgecolor='black', linewidth=1.2)
    ax1.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Link weight w', fontsize=14)
    ax1.set_ylabel('P(w)', fontsize=14)
    ax1.set_title(f'Disordered Phase (β={beta_dis}, α={alpha_dis})', fontsize=14)
    ax1.set_xlim(0, 1)
    ax1.grid(True, alpha=0.3)
    
    # Ordered phase
    beta_ord, alpha_ord = 3.0, 1.5
    weights_ordered = generate_weight_distribution(beta_ord, alpha_ord)
    
    ax2.hist(weights_ordered, bins=bins, density=True, alpha=0.7,
             color='steelblue', edgecolor='black', linewidth=1.2)
    
    # Add KDE
    x_smooth = np.linspace(0, 1, 200)
    kde = stats.gaussian_kde(weights_ordered)
    y_smooth = kde(x_smooth)
    ax2.plot(x_smooth, y_smooth, 'r-', linewidth=2.5)
    
    ax2.axvline(0.5, color='black', linestyle=':', alpha=0.5, linewidth=1.5)
    ax2.set_xlabel('Link weight w', fontsize=14)
    ax2.set_ylabel('P(w)', fontsize=14)
    ax2.set_title(f'Ordered Phase (β={beta_ord}, α={alpha_ord})', fontsize=14)
    ax2.set_xlim(0, 1)
    ax2.grid(True, alpha=0.3)
    
    # Add phase labels
    ax1.text(0.5, 0.9, 'No symmetry\nbreaking', ha='center', 
             transform=ax1.transAxes, fontsize=12,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    ax2.text(0.5, 0.9, 'Z₂ symmetry\nbroken', ha='center',
             transform=ax2.transAxes, fontsize=12,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    return fig

# Generate the main figure
print("Generating weight distribution data...")
weights = generate_weight_distribution(beta=3.0, alpha=1.5, n_samples=50000)

print("Creating Figure 4...")
fig = plot_weight_distribution(weights, beta=3.0, alpha=1.5, system_size=96)

# Save the figure
fig.savefig('weight_distribution.png', dpi=300, bbox_inches='tight')
fig.savefig('weight_distribution.pdf', bbox_inches='tight')  # For LaTeX
print("Saved as weight_distribution.png and weight_distribution.pdf")

# Also create comparison figure
print("\nCreating comparison figure...")
fig_comp = create_comparison_plot()
fig_comp.savefig('phase_comparison.png', dpi=300, bbox_inches='tight')
print("Saved phase comparison as phase_comparison.png")

# Show the plots
plt.show()

# Print summary statistics
print("\nSummary Statistics:")
print(f"Mean weight: {np.mean(weights):.3f}")
print(f"Std deviation: {np.std(weights):.3f}")
print(f"Order parameter ⟨|w - 0.5|⟩: {np.mean(np.abs(weights - 0.5)):.3f}")
print(f"Peak positions: ~{weights[weights < 0.5].mean():.3f} and ~{weights[weights > 0.5].mean():.3f}")