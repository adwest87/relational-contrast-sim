#!/usr/bin/env python3
"""
Create a focused 2D figure showing ridge alignment with enhanced visualization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from datetime import datetime

# Set publication quality defaults
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

def load_all_data():
    """Load data for all system sizes."""
    
    data_files = {
        24: ['fss_data/results_n24.csv'],
        48: ['fss_data/results_n48.csv'],
        96: ['fss_data/results_n96.csv', 'fss_data/results_n96_critical.csv']
    }
    
    all_data = {}
    
    for N, files in data_files.items():
        system_data = []
        for file in files:
            try:
                df = pd.read_csv(file)
                if 'n_nodes' in df.columns:
                    df = df[df['n_nodes'] == N]
                system_data.append(df)
            except:
                continue
        
        if system_data:
            combined = pd.concat(system_data, ignore_index=True)
            all_data[N] = combined.groupby(['beta', 'alpha']).agg({
                'susceptibility': ['mean', 'std', 'count']
            }).reset_index()
            all_data[N].columns = ['beta', 'alpha', 'chi_mean', 'chi_std', 'count']
    
    return all_data

def create_enhanced_ridge_plot():
    """Create enhanced 2D ridge alignment plot."""
    
    # Load data
    all_data = load_all_data()
    
    # Peak locations and colors
    peaks = {
        24: (2.900, 1.490),
        48: (2.910, 1.480),
        96: (2.930, 1.470)
    }
    
    colors = {
        24: '#3498db',  # Blue
        48: '#e74c3c',  # Red
        96: '#27ae60'   # Green
    }
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot susceptibility contours for each system size
    for N, data in all_data.items():
        # Get high susceptibility points
        max_chi = data['chi_mean'].max()
        
        # Plot points with susceptibility-based sizing
        sizes = 50 * (data['chi_mean'] / max_chi)**2  # Square for more contrast
        scatter = ax.scatter(data['beta'], data['alpha'], 
                           s=sizes, c=data['chi_mean'], 
                           cmap='viridis', alpha=0.5,
                           edgecolor=colors[N], linewidth=0.5)
        
        # Add contour ellipses around high-chi regions
        high_chi = data[data['chi_mean'] > 0.7 * max_chi]
        if len(high_chi) > 3:
            # Calculate covariance for ellipse
            cov = np.cov(high_chi['beta'], high_chi['alpha'])
            
            # Get ellipse parameters
            lambda_, v = np.linalg.eig(cov)
            lambda_ = np.sqrt(lambda_)
            angle = np.degrees(np.arctan2(v[1, 0], v[0, 0]))
            
            # Plot ellipse
            ellipse = Ellipse((peaks[N][0], peaks[N][1]),
                            width=2*lambda_[0], height=2*lambda_[1],
                            angle=angle, facecolor='none',
                            edgecolor=colors[N], linewidth=2,
                            linestyle='--', alpha=0.7)
            ax.add_patch(ellipse)
    
    # Plot ridge line from combined fit
    beta_all = []
    alpha_all = []
    chi_all = []
    
    for N, data in all_data.items():
        high_chi = data[data['chi_mean'] > 0.7 * data['chi_mean'].max()]
        beta_all.extend(high_chi['beta'].values)
        alpha_all.extend(high_chi['alpha'].values)
        chi_all.extend(high_chi['chi_mean'].values)
    
    # Weighted fit
    weights = np.array(chi_all) / np.sum(chi_all)
    coeffs = np.polyfit(beta_all, alpha_all, 1, w=weights)
    
    # Plot ridge line
    beta_range = np.linspace(2.85, 2.95, 100)
    alpha_ridge = coeffs[0] * beta_range + coeffs[1]
    ax.plot(beta_range, alpha_ridge, 'k-', linewidth=3, 
           label=f'Ridge: α = {coeffs[0]:.3f}β + {coeffs[1]:.3f}', zorder=5)
    
    # Add uncertainty band
    residuals = np.array(alpha_all) - (coeffs[0] * np.array(beta_all) + coeffs[1])
    std_resid = np.std(residuals)
    ax.fill_between(beta_range, 
                   alpha_ridge - std_resid, 
                   alpha_ridge + std_resid,
                   color='gray', alpha=0.2, label='Ridge uncertainty')
    
    # Plot peak positions
    for N, (beta, alpha) in peaks.items():
        ax.scatter(beta, alpha, s=400, marker='*', 
                  color=colors[N], edgecolor='black', linewidth=2,
                  zorder=10, label=f'N={N} peak')
        
        # Add annotations with boxes
        ax.annotate(f'N={N}\n({beta:.3f}, {alpha:.3f})', 
                   (beta, alpha), xytext=(10, 10),
                   textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.5', 
                           facecolor=colors[N], alpha=0.8, edgecolor='black'),
                   color='white', fontweight='bold', fontsize=10,
                   ha='left')
    
    # Add flow arrows
    for i, N1 in enumerate([24, 48]):
        N2 = [48, 96][i]
        ax.annotate('', xy=peaks[N2], xytext=peaks[N1],
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2',
                                 color='red', linewidth=2.5, alpha=0.8))
    
    # Extrapolated infinite-volume point
    beta_inf = 2.935
    alpha_inf = coeffs[0] * beta_inf + coeffs[1]
    ax.scatter(beta_inf, alpha_inf, s=300, marker='D', 
              color='purple', edgecolor='black', linewidth=2,
              zorder=10, label='N→∞ extrapolation')
    
    # Format plot
    ax.set_xlabel('β (weight coupling)', fontsize=14)
    ax.set_ylabel('α (trace weight)', fontsize=14)
    ax.set_title('Critical Ridge Alignment Across System Sizes\n' + 
                'High-χ regions and peak evolution', fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(2.85, 0.95)
    ax.set_ylim(1.45, 1.52)
    
    # Legend
    ax.legend(loc='lower left', frameon=True, fancybox=True, 
             framealpha=0.9, fontsize=10)
    
    # Add inset with peak evolution
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    axins = inset_axes(ax, width="35%", height="35%", loc='upper right')
    
    # Plot peak trajectory
    N_values = [24, 48, 96]
    beta_peaks = [peaks[N][0] for N in N_values]
    alpha_peaks = [peaks[N][1] for N in N_values]
    
    axins.plot(N_values, beta_peaks, 'o-', color='navy', 
              linewidth=2, markersize=8, label='β(N)')
    axins.plot(N_values, alpha_peaks, 's-', color='darkred', 
              linewidth=2, markersize=8, label='α(N)')
    
    axins.set_xlabel('System size N', fontsize=10)
    axins.set_ylabel('Parameter value', fontsize=10)
    axins.set_title('Peak Evolution', fontsize=11)
    axins.grid(True, alpha=0.3)
    axins.legend(fontsize=9)
    
    # Save figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'ridge_alignment_enhanced_{timestamp}.png'
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"Saved: {filename}")
    
    # Also save without timestamp
    plt.savefig('ridge_alignment.png', dpi=300)
    
    return coeffs[0], coeffs[1]

def main():
    """Create enhanced ridge alignment figure."""
    
    print("="*60)
    print("Creating Enhanced Ridge Alignment Figure")
    print("="*60)
    
    slope, intercept = create_enhanced_ridge_plot()
    
    print(f"\nRidge equation: α = {slope:.3f}β + {intercept:.3f}")
    print(f"Slope interpretation: dα/dβ = {slope:.3f}")
    print("\nThis means for every unit increase in β,")
    print(f"α decreases by {-slope:.3f} to maintain criticality.")
    print("="*60)

if __name__ == "__main__":
    main()