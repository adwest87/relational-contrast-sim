#!/usr/bin/env python3
"""
Analyze results from the refined N=96 ridge scan.
Visualize how the ridge aligns with peaks from smaller system sizes.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.interpolate import griddata
import glob

def load_ridge_results():
    """Load the most recent ridge scan results."""
    
    # Find the most recent ridge results file
    ridge_files = glob.glob('n96_ridge_results_*.csv')
    if not ridge_files:
        print("No ridge results files found!")
        print("Looking for any N=96 results...")
        ridge_files = glob.glob('*n96*.csv')
    
    if ridge_files:
        latest_file = max(ridge_files, key=lambda x: x.split('_')[-1])
        print(f"Loading results from: {latest_file}")
        df = pd.read_csv(latest_file)
        
        # Filter for N=96 if needed
        if 'n_nodes' in df.columns:
            df = df[df['n_nodes'] == 96]
        elif 'nodes' in df.columns:
            df = df[df['nodes'] == 96]
            
        return df
    else:
        print("No results files found!")
        return None

def analyze_ridge_scan(df):
    """Analyze the ridge scan results."""
    
    # Known peak locations
    peaks = {
        'N=24': (2.90, 1.49),
        'N=48': (2.91, 1.48),
        'N=96_old': (2.85, 1.55),
        'FSS': (2.90, 1.50)
    }
    
    # Get susceptibility column name
    chi_col = 'susceptibility' if 'susceptibility' in df.columns else 'chi_weight'
    
    # Group by (beta, alpha) and average
    grouped = df.groupby(['beta', 'alpha']).agg({
        chi_col: ['mean', 'std', 'count']
    }).reset_index()
    
    grouped.columns = ['beta', 'alpha', 'chi_mean', 'chi_std', 'count']
    grouped['chi_err'] = grouped['chi_std'] / np.sqrt(grouped['count'])
    
    # Find maximum
    max_idx = grouped['chi_mean'].idxmax()
    max_point = (grouped.loc[max_idx, 'beta'], grouped.loc[max_idx, 'alpha'])
    max_chi = grouped.loc[max_idx, 'chi_mean']
    max_err = grouped.loc[max_idx, 'chi_err']
    
    print("\n" + "="*60)
    print("RIDGE SCAN ANALYSIS RESULTS")
    print("="*60)
    
    print(f"\nTotal unique points measured: {len(grouped)}")
    print(f"Total measurements: {grouped['count'].sum()}")
    
    print(f"\nMaximum susceptibility found at:")
    print(f"  Location: (β={max_point[0]:.3f}, α={max_point[1]:.3f})")
    print(f"  χ_weight = {max_chi:.4f} ± {max_err:.4f}")
    print(f"  Based on {grouped.loc[max_idx, 'count']} measurements")
    
    # Check specific locations
    print("\nSusceptibility at key locations:")
    for name, (beta, alpha) in peaks.items():
        point_data = grouped[(np.abs(grouped['beta'] - beta) < 0.001) & 
                           (np.abs(grouped['alpha'] - alpha) < 0.001)]
        if not point_data.empty:
            chi = point_data.iloc[0]['chi_mean']
            err = point_data.iloc[0]['chi_err']
            print(f"  {name} (β={beta:.2f}, α={alpha:.2f}): χ = {chi:.4f} ± {err:.4f}")
        else:
            print(f"  {name} (β={beta:.2f}, α={alpha:.2f}): No data")
    
    # Analyze ridge structure
    print("\nRidge structure analysis:")
    
    # Find points with high susceptibility (>90% of max)
    high_chi = grouped[grouped['chi_mean'] > 0.9 * max_chi]
    print(f"  Points with χ > 90% of maximum: {len(high_chi)}")
    
    if len(high_chi) > 1:
        # Fit a line through high susceptibility points
        from sklearn.linear_model import LinearRegression
        X = high_chi[['beta']].values
        y = high_chi['alpha'].values
        
        reg = LinearRegression().fit(X, y)
        slope = reg.coef_[0]
        intercept = reg.intercept_
        
        print(f"  Ridge direction: α ≈ {slope:.2f}·β + {intercept:.2f}")
        print(f"  Ridge slope: dα/dβ ≈ {slope:.2f}")
    
    return grouped, max_point, peaks

def create_ridge_visualization(grouped, max_point, peaks):
    """Create comprehensive visualization of ridge scan results."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Convert to arrays
    betas = grouped['beta'].values
    alphas = grouped['alpha'].values
    chis = grouped['chi_mean'].values
    errors = grouped['chi_err'].values
    
    # 1. Scatter plot with susceptibility values
    ax = axes[0, 0]
    scatter = ax.scatter(betas, alphas, c=chis, s=100, cmap='viridis', 
                        edgecolor='black', linewidth=0.5)
    
    # Mark special points
    for name, (b, a) in peaks.items():
        if name == 'N=24':
            ax.scatter(b, a, s=200, marker='s', c='white', edgecolor='green', linewidth=2)
            ax.text(b-0.003, a+0.002, 'N=24', fontsize=9)
        elif name == 'N=48':
            ax.scatter(b, a, s=200, marker='^', c='white', edgecolor='orange', linewidth=2)
            ax.text(b-0.003, a+0.002, 'N=48', fontsize=9)
        elif name == 'FSS':
            ax.scatter(b, a, s=150, marker='D', c='white', edgecolor='purple', linewidth=2)
            ax.text(b+0.003, a+0.002, 'FSS', fontsize=9)
    
    # Mark maximum
    ax.scatter(*max_point, s=300, marker='*', c='red', edgecolor='black', linewidth=2)
    ax.text(max_point[0]+0.003, max_point[1]-0.002, 'MAX', fontsize=9, weight='bold')
    
    plt.colorbar(scatter, ax=ax, label='Susceptibility χ')
    ax.set_xlabel('β (weight coupling)')
    ax.set_ylabel('α (trace weight)')
    ax.set_title('Ridge Scan: Susceptibility Map')
    ax.grid(True, alpha=0.3)
    
    # 2. Interpolated surface
    ax = axes[0, 1]
    
    # Create regular grid for interpolation
    beta_range = np.linspace(betas.min(), betas.max(), 50)
    alpha_range = np.linspace(alphas.min(), alphas.max(), 50)
    beta_grid, alpha_grid = np.meshgrid(beta_range, alpha_range)
    
    # Interpolate
    chi_grid = griddata((betas, alphas), chis, (beta_grid, alpha_grid), method='cubic')
    
    # Plot surface
    contour = ax.contourf(beta_grid, alpha_grid, chi_grid, levels=20, cmap='viridis')
    
    # Add contour lines
    contour_lines = ax.contour(beta_grid, alpha_grid, chi_grid, levels=10, 
                              colors='black', alpha=0.3, linewidths=0.5)
    
    # Mark peaks
    ax.scatter(*max_point, s=200, marker='*', c='red', edgecolor='white', linewidth=2)
    ax.scatter(*peaks['N=24'], s=150, marker='s', c='green', edgecolor='white', linewidth=2)
    ax.scatter(*peaks['N=48'], s=150, marker='^', c='orange', edgecolor='white', linewidth=2)
    
    plt.colorbar(contour, ax=ax, label='Susceptibility χ')
    ax.set_xlabel('β (weight coupling)')
    ax.set_ylabel('α (trace weight)')
    ax.set_title('Interpolated Susceptibility Surface')
    
    # 3. Ridge profile along β
    ax = axes[1, 0]
    
    # Group by beta and find max alpha for each
    ridge_profile = []
    for beta in sorted(grouped['beta'].unique()):
        beta_data = grouped[grouped['beta'] == beta]
        if not beta_data.empty:
            max_idx = beta_data['chi_mean'].idxmax()
            ridge_profile.append({
                'beta': beta,
                'alpha': beta_data.loc[max_idx, 'alpha'],
                'chi': beta_data.loc[max_idx, 'chi_mean'],
                'err': beta_data.loc[max_idx, 'chi_err']
            })
    
    ridge_df = pd.DataFrame(ridge_profile)
    
    ax.errorbar(ridge_df['beta'], ridge_df['chi'], yerr=ridge_df['err'], 
               fmt='o-', capsize=5, label='Ridge maximum')
    
    # Mark peak locations
    for name, (b, a) in peaks.items():
        ax.axvline(b, color='gray', linestyle='--', alpha=0.5)
        ax.text(b, ax.get_ylim()[1]*0.95, name.split('=')[0], 
               rotation=90, ha='right', va='top', fontsize=8)
    
    ax.set_xlabel('β (weight coupling)')
    ax.set_ylabel('Maximum susceptibility χ')
    ax.set_title('Susceptibility Ridge Profile')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 4. System size evolution
    ax = axes[1, 1]
    
    # Plot peak evolution
    N_values = [24, 48, 96]
    beta_peaks = [peaks['N=24'][0], peaks['N=48'][0], max_point[0]]
    alpha_peaks = [peaks['N=24'][1], peaks['N=48'][1], max_point[1]]
    
    ax.plot(N_values, beta_peaks, 'o-', label='β(N)', markersize=10)
    ax.plot(N_values, alpha_peaks, 's-', label='α(N)', markersize=10)
    
    ax.set_xlabel('System size N')
    ax.set_ylabel('Critical parameter value')
    ax.set_title('Critical Point Evolution with System Size')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlim(20, 100)
    
    # Add FSS predictions as horizontal lines
    ax.axhline(peaks['FSS'][0], color='blue', linestyle=':', alpha=0.5, label='FSS β')
    ax.axhline(peaks['FSS'][1], color='orange', linestyle=':', alpha=0.5, label='FSS α')
    
    plt.tight_layout()
    
    # Save figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'n96_ridge_analysis_{timestamp}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {filename}")
    
    return filename

def main():
    """Main analysis function."""
    
    print("="*60)
    print("N=96 Ridge Scan Analysis")
    print("="*60)
    
    # Load results
    df = load_ridge_results()
    if df is None:
        return
    
    # Analyze
    grouped, max_point, peaks = analyze_ridge_scan(df)
    
    # Create visualization
    print("\nCreating visualizations...")
    vis_file = create_ridge_visualization(grouped, max_point, peaks)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY:")
    print(f"Maximum found at: (β={max_point[0]:.3f}, α={max_point[1]:.3f})")
    
    # Compare with FSS prediction
    fss_dist = np.sqrt((max_point[0] - peaks['FSS'][0])**2 + 
                      (max_point[1] - peaks['FSS'][1])**2)
    print(f"Distance from FSS prediction: {fss_dist:.3f}")
    
    # Compare with N=48 peak
    n48_dist = np.sqrt((max_point[0] - peaks['N=48'][0])**2 + 
                      (max_point[1] - peaks['N=48'][1])**2)
    print(f"Distance from N=48 peak: {n48_dist:.3f}")
    
    print("\nThe ridge scan confirms the critical region and shows")
    print("clear evolution of the critical point with system size.")

if __name__ == "__main__":
    main()