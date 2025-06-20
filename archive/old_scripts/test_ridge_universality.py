#!/usr/bin/env python3
"""
Test universality class along the critical ridge for N=48.
Sample 5 points along α = 0.060β + 1.313 and measure susceptibility and Binder cumulant.
"""

import numpy as np
import pandas as pd
import subprocess
import os
from datetime import datetime

def generate_ridge_points(N=48):
    """Generate points along the critical ridge."""
    
    # Ridge equation: α = 0.060β + 1.313
    slope = 0.060
    intercept = 1.313
    
    # 5 points with β ranging from 2.85 to 2.95
    beta_values = np.linspace(2.85, 2.95, 5)
    
    points = []
    for beta in beta_values:
        alpha = slope * beta + intercept
        points.append((beta, alpha))
    
    return points, slope, intercept

def create_points_file(points, filename='ridge_universality_points.csv'):
    """Create CSV file with ridge points."""
    
    with open(filename, 'w') as f:
        for beta, alpha in points:
            f.write(f"{beta:.3f},{alpha:.3f}\n")
    
    return filename

def create_run_script(points, N=48):
    """Create shell script to run the simulations."""
    
    script_content = f"""#!/bin/bash
# Test universality along critical ridge for N={N}
# Generated: {datetime.now()}

echo "=================================================="
echo "TESTING UNIVERSALITY ALONG CRITICAL RIDGE"
echo "=================================================="
echo ""
echo "System size: N={N}"
echo "Ridge equation: α = 0.060β + 1.313"
echo "Testing {len(points)} points along ridge"
echo ""

# Run parameters
STEPS=500000      # 5×10^5 MC steps
REPLICAS=20       # 20 replicas for good statistics
EQUILIBRATION=100000  # 20% for equilibration

echo "Parameters:"
echo "  MC steps: $STEPS"
echo "  Replicas: $REPLICAS"
echo "  Equilibration: $EQUILIBRATION"
echo ""

# Show points being tested
echo "Points to test:"
"""
    
    for i, (beta, alpha) in enumerate(points):
        script_content += f'echo "  {i+1}. β={beta:.3f}, α={alpha:.3f}"\n'
    
    script_content += """
echo ""
echo "Starting simulations..."
echo ""

# Run the scan
cargo run --release --bin fss_narrow_scan -- \\
  --pairs ridge_universality_points.csv \\
  --output ridge_universality_N48_$(date +%Y%m%d_%H%M%S).csv \\
  --nodes 48 \\
  --steps $STEPS \\
  --replicas $REPLICAS \\
  --debug

echo ""
echo "Simulations complete!"
echo ""
echo "To analyze results, run:"
echo "  python3 scripts/analyze_ridge_universality.py"
"""
    
    filename = 'run_ridge_universality_test.sh'
    with open(filename, 'w') as f:
        f.write(script_content)
    
    os.chmod(filename, 0o755)
    return filename

def create_analysis_script():
    """Create script to analyze universality results."""
    
    script_content = '''#!/usr/bin/env python3
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
    
    print("\\n" + "="*60)
    print("UNIVERSALITY ANALYSIS ALONG CRITICAL RIDGE")
    print("="*60)
    print(f"\\nRidge equation: α = 0.060β + 1.313")
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
        print(f"\\nBinder cumulant variance along ridge: {binder_variance:.6f}")
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
        print(f"\\nSusceptibility ratio variance: {ratio_variance:.6f}")
    
    plt.tight_layout()
    
    # Save figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'ridge_universality_analysis_{timestamp}.png'
    plt.savefig(filename, dpi=300)
    print(f"\\nFigure saved: {filename}")
    
    # Summary statistics
    print("\\n" + "="*60)
    print("SUMMARY:")
    print("="*60)
    
    print("\\nPoint-by-point results:")
    for _, row in grouped.iterrows():
        print(f"\\nβ={row['beta']:.3f}, α={row['alpha']:.3f}:")
        print(f"  χ = {row['susceptibility_mean']:.2f} ± {row['susceptibility_std']/np.sqrt(row['susceptibility_count']):.2f}")
        if 'binder_mean' in row:
            print(f"  U₄ = {row['binder_mean']:.4f} ± {row['binder_std']/np.sqrt(row['binder_count']):.4f}")
    
    # Test for consistency
    print("\\n" + "-"*60)
    print("UNIVERSALITY TEST:")
    
    # Check if Binder cumulant is approximately constant
    if 'binder_mean' in grouped.columns:
        binder_std_total = np.std(binder_mean)
        binder_mean_total = np.mean(binder_mean)
        cv = binder_std_total / binder_mean_total if binder_mean_total > 0 else np.inf
        
        print(f"\\nBinder cumulant along ridge:")
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
    print(f"\\nSusceptibility variation:")
    chi_cv = np.std(chi_mean) / np.mean(chi_mean)
    print(f"  Coefficient of variation: {chi_cv:.4f}")
    
    if chi_cv > 0.5:
        print("  → Large variation expected (different effective system sizes)")
    
    print("\\n" + "="*60)
    
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
    print(f"\\nNumerical summary saved: ridge_universality_summary.csv")

if __name__ == "__main__":
    main()
'''
    
    with open('scripts/analyze_ridge_universality.py', 'w') as f:
        f.write(script_content)
    
    os.chmod('scripts/analyze_ridge_universality.py', 0o755)
    return 'scripts/analyze_ridge_universality.py'

def main():
    """Generate ridge universality test."""
    
    print("="*60)
    print("RIDGE UNIVERSALITY TEST SETUP")
    print("="*60)
    
    # Generate points
    points, slope, intercept = generate_ridge_points(N=48)
    
    print(f"\nRidge equation: α = {slope:.3f}β + {intercept:.3f}")
    print(f"\nGenerating {len(points)} points for N=48:")
    print("-"*40)
    
    for i, (beta, alpha) in enumerate(points):
        print(f"{i+1}. β = {beta:.3f}, α = {alpha:.3f}")
    
    # Create points file
    points_file = create_points_file(points)
    print(f"\nPoints saved to: {points_file}")
    
    # Create run script
    run_script = create_run_script(points)
    print(f"Run script created: {run_script}")
    
    # Create analysis script
    analysis_script = create_analysis_script()
    print(f"Analysis script created: {analysis_script}")
    
    print("\n" + "="*60)
    print("SETUP COMPLETE")
    print("="*60)
    print("\nTo run the test:")
    print(f"  ./{run_script}")
    print("\nThis will:")
    print("  - Run 500,000 MC steps per point")
    print("  - Use 20 replicas for statistics")
    print("  - Test 5 points along the critical ridge")
    print("\nEstimated runtime: ~30-60 minutes")
    print("\nThe analysis will determine if:")
    print("  - Binder cumulant is constant (same universality class)")
    print("  - Or varies along ridge (changing universality)")
    print("="*60)

if __name__ == "__main__":
    main()