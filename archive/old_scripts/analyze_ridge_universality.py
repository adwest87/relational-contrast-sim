#!/usr/bin/env python3
"""
Analyze universality class along the critical ridge.
Check if Binder cumulant and susceptibility scaling are consistent.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from datetime import datetime

def load_ridge_results():
    """Load results from ridge universality test."""
    
    # Find most recent results file
    files = glob.glob('ridge_universality_N48_*.csv')
    if not files:
        print("No results files found!")
        return None
    
    latest = max(files, key=lambda x: x.split('_')[-1])
    print(f"Loading results from: {latest}")
    
    df = pd.read_csv(latest)
    
    # Filter for N=48 if needed
    if 'n_nodes' in df.columns:
        df = df[df['n_nodes'] == 48]
    elif 'nodes' in df.columns:
        df = df[df['nodes'] == 48]
    
    return df

def analyze_universality(df):
    """Analyze universality indicators along ridge."""
    
    # Group by (beta, alpha) and compute statistics
    grouped = df.groupby(['beta', 'alpha']).agg({
        'susceptibility': ['mean', 'std', 'count'],
        'binder': ['mean', 'std', 'count'] if 'binder' in df.columns else ['mean'],
        'mean_action': ['mean', 'std'] if 'mean_action' in df.columns else ['mean']
    }).reset_index()
    
    # Flatten column names
    grouped.columns = ['_'.join(col).strip('_') for col in grouped.columns.values]
    
    # Sort by beta
    grouped = grouped.sort_values('beta')
    
    print("\n" + "="*60)
    print("UNIVERSALITY ANALYSIS ALONG CRITICAL RIDGE")
    print("="*60)
    print(f"\nRidge equation: α = 0.060β + 1.313")
    print(f"Number of points: {len(grouped)}")
    
    # Create figure with multiple panels
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Susceptibility along ridge
    ax1 = axes[0, 0]
    chi_mean = grouped['susceptibility_mean'].values
    chi_err = grouped['susceptibility_std'].values / np.sqrt(grouped['susceptibility_count'].values)
    
    ax1.errorbar(grouped['beta'], chi_mean, yerr=chi_err, 
                fmt='o-', markersize=8, capsize=5)
    ax1.set_xlabel('β')
    ax1.set_ylabel('Susceptibility χ')
    ax1.set_title('Susceptibility Along Ridge')
    ax1.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(grouped['beta'], chi_mean, 1)
    p = np.poly1d(z)
    ax1.plot(grouped['beta'], p(grouped['beta']), 'r--', alpha=0.5,
            label=f'Trend: {z[0]:.1f}β + {z[1]:.1f}')
    ax1.legend()
    
    # 2. Binder cumulant along ridge
    ax2 = axes[0, 1]
    if 'binder_mean' in grouped.columns:
        binder_mean = grouped['binder_mean'].values
        binder_err = grouped['binder_std'].values / np.sqrt(grouped['binder_count'].values)
        
        ax2.errorbar(grouped['beta'], binder_mean, yerr=binder_err,
                    fmt='s-', markersize=8, capsize=5, color='green')
        ax2.set_xlabel('β')
        ax2.set_ylabel('Binder Cumulant U₄')
        ax2.set_title('Binder Cumulant Along Ridge')
        ax2.grid(True, alpha=0.3)
        
        # Reference lines for universality classes
        ax2.axhline(0.611, color='blue', linestyle='--', alpha=0.5, label='2D Ising')
        ax2.axhline(0.465, color='red', linestyle='--', alpha=0.5, label='3D Ising')
        ax2.legend()
        
        # Calculate variance
        binder_variance = np.var(binder_mean)
        print(f"\nBinder cumulant variance along ridge: {binder_variance:.6f}")
        print(f"Binder range: [{np.min(binder_mean):.4f}, {np.max(binder_mean):.4f}]")
    
    # 3. Phase space view
    ax3 = axes[1, 0]
    scatter = ax3.scatter(grouped['beta'], grouped['alpha'], 
                         c=chi_mean, s=100, cmap='viridis')
    
    # Plot ridge line
    beta_range = np.linspace(grouped['beta'].min(), grouped['beta'].max(), 100)
    alpha_ridge = 0.060 * beta_range + 1.313
    ax3.plot(beta_range, alpha_ridge, 'r--', linewidth=2, label='Ridge')
    
    ax3.set_xlabel('β')
    ax3.set_ylabel('α') 
    ax3.set_title('Points Along Ridge')
    plt.colorbar(scatter, ax=ax3, label='χ')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Susceptibility ratios (test for constant γ/ν)
    ax4 = axes[1, 1]
    if len(chi_mean) > 1:
        # Calculate ratios between consecutive points
        chi_ratios = chi_mean[1:] / chi_mean[:-1]
        beta_mids = (grouped['beta'].values[1:] + grouped['beta'].values[:-1]) / 2
        
        ax4.plot(beta_mids, chi_ratios, 'o-', markersize=8)
        ax4.set_xlabel('β (midpoint)')
        ax4.set_ylabel('χ(i+1) / χ(i)')
        ax4.set_title('Susceptibility Ratios')
        ax4.grid(True, alpha=0.3)
        
        # If ratios are constant, universality class is same
        ratio_variance = np.var(chi_ratios)
        print(f"\nSusceptibility ratio variance: {ratio_variance:.6f}")
    
    plt.tight_layout()
    
    # Save figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'ridge_universality_analysis_{timestamp}.png'
    plt.savefig(filename, dpi=300)
    print(f"\nFigure saved: {filename}")
    
    # Summary statistics
    print("\n" + "="*60)
    print("SUMMARY:")
    print("="*60)
    
    print("\nPoint-by-point results:")
    for _, row in grouped.iterrows():
        print(f"\nβ={row['beta']:.3f}, α={row['alpha']:.3f}:")
        print(f"  χ = {row['susceptibility_mean']:.2f} ± {row['susceptibility_std']/np.sqrt(row['susceptibility_count']):.2f}")
        if 'binder_mean' in row:
            print(f"  U₄ = {row['binder_mean']:.4f} ± {row['binder_std']/np.sqrt(row['binder_count']):.4f}")
    
    # Test for consistency
    print("\n" + "-"*60)
    print("UNIVERSALITY TEST:")
    
    # Check if Binder cumulant is approximately constant
    if 'binder_mean' in grouped.columns:
        binder_std_total = np.std(binder_mean)
        binder_mean_total = np.mean(binder_mean)
        cv = binder_std_total / binder_mean_total if binder_mean_total > 0 else np.inf
        
        print(f"\nBinder cumulant along ridge:")
        print(f"  Mean: {binder_mean_total:.4f}")
        print(f"  Std Dev: {binder_std_total:.4f}")
        print(f"  Coefficient of variation: {cv:.4f}")
        
        if cv < 0.05:
            print("  → Binder cumulant is CONSISTENT along ridge (CV < 5%)")
            print("  → Suggests SAME universality class")
        elif cv < 0.10:
            print("  → Binder cumulant shows SMALL variation (CV < 10%)")
            print("  → Likely same universality class with corrections")
        else:
            print("  → Binder cumulant shows SIGNIFICANT variation (CV > 10%)")
            print("  → May indicate VARYING universality class")
    
    # Check susceptibility scaling
    print(f"\nSusceptibility variation:")
    chi_cv = np.std(chi_mean) / np.mean(chi_mean)
    print(f"  Coefficient of variation: {chi_cv:.4f}")
    
    if chi_cv > 0.5:
        print("  → Large variation expected (different effective system sizes)")
    
    print("\n" + "="*60)
    
    return grouped

def main():
    """Run the analysis."""
    
    # Load results
    df = load_ridge_results()
    if df is None:
        return
    
    # Analyze
    results = analyze_universality(df)
    
    # Save numerical results
    results.to_csv('ridge_universality_summary.csv', index=False)
    print(f"\nNumerical summary saved: ridge_universality_summary.csv")

if __name__ == "__main__":
    main()
