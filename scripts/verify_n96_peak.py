#!/usr/bin/env python3
"""
Quick verification scan for N=96 to check if the true susceptibility peak
is near (β=2.90, α=1.50) where finite-size scaling analysis suggests.

Creates a 3×3 grid centered at (β=2.90, α=1.50) with spacing 0.01,
plus checks the old peak location (β=2.85, α=1.55) for comparison.
"""

import numpy as np
import pandas as pd
import subprocess
import time
from datetime import datetime
import os

def run_verification_scan():
    """Run quick verification scan for N=96 susceptibility peak."""
    
    # Parameters
    N = 96
    num_steps = 100000  # 1×10^5 MC steps
    num_replicas = 5    # Multiple replicas for better statistics
    
    # Create 3×3 grid around (β=2.90, α=1.50)
    beta_center, alpha_center = 2.90, 1.50
    spacing = 0.01
    
    beta_values = [beta_center - spacing, beta_center, beta_center + spacing]
    alpha_values = [alpha_center - spacing, alpha_center, alpha_center + spacing]
    
    # Also check old peak location
    old_peak = (2.85, 1.55)
    
    print(f"=== N=96 Susceptibility Peak Verification Scan ===")
    print(f"Time: {datetime.now()}")
    print(f"MC steps per point: {num_steps:,}")
    print(f"Replicas per point: {num_replicas}")
    print(f"\nScanning 3×3 grid around (β={beta_center}, α={alpha_center})")
    print(f"β range: [{beta_values[0]:.2f}, {beta_values[-1]:.2f}]")
    print(f"α range: [{alpha_values[0]:.2f}, {alpha_values[-1]:.2f}]")
    print(f"Also checking old peak at (β={old_peak[0]}, α={old_peak[1]})")
    print("\n" + "="*60 + "\n")
    
    # Create points file
    points_file = 'verification_points.csv'
    with open(points_file, 'w') as f:
        # Grid points
        for beta in beta_values:
            for alpha in alpha_values:
                f.write(f"{beta:.2f},{alpha:.2f}\n")
        # Old peak
        f.write(f"{old_peak[0]:.2f},{old_peak[1]:.2f}\n")
    
    total_points = len(beta_values) * len(alpha_values) + 1
    print(f"Created {points_file} with {total_points} points")
    
    # Run simulation using Rust binary
    output_file = f'verification_N{N}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    
    cmd = [
        'cargo', 'run', '--release', '--bin', 'fss_narrow_scan', '--',
        '--pairs', points_file,
        '--output', output_file,
        '--nodes', str(N),
        '--steps', str(num_steps),
        '--replicas', str(num_replicas),
        '--debug'
    ]
    
    print(f"\nRunning simulation...")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 60)
    
    start_time = time.time()
    
    try:
        # Run the simulation
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(result.stdout)
        
    except subprocess.CalledProcessError as e:
        print(f"Error running simulation: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return
    
    elapsed = time.time() - start_time
    print(f"\nSimulation completed in {elapsed:.1f}s")
    
    # Read and analyze results
    print(f"\n{'='*60}")
    print("\n=== RESULTS SUMMARY ===\n")
    
    try:
        # Read the CSV output
        df = pd.read_csv(output_file)
        
        # Group by (beta, alpha) and average over replicas
        grouped = df.groupby(['beta', 'alpha']).agg({
            'chi_weight': ['mean', 'std', 'count']
        }).reset_index()
        
        # Flatten column names
        grouped.columns = ['beta', 'alpha', 'chi_mean', 'chi_std', 'count']
        
        # Calculate standard error
        grouped['chi_err'] = grouped['chi_std'] / np.sqrt(grouped['count'])
        
        # Separate grid points from old peak
        grid_data = grouped[grouped.apply(lambda row: 
            (row['beta'], row['alpha']) != old_peak, axis=1)]
        old_data = grouped[grouped.apply(lambda row: 
            (row['beta'], row['alpha']) == old_peak, axis=1)]
        
        # Find maximum in grid
        max_idx = grid_data['chi_mean'].idxmax()
        max_point = (grid_data.loc[max_idx, 'beta'], grid_data.loc[max_idx, 'alpha'])
        max_chi = grid_data.loc[max_idx, 'chi_mean']
        max_chi_err = grid_data.loc[max_idx, 'chi_err']
        
        # Get old peak data
        if not old_data.empty:
            chi_old = old_data.iloc[0]['chi_mean']
            chi_old_err = old_data.iloc[0]['chi_err']
        else:
            print("Warning: No data found for old peak location!")
            chi_old = chi_old_err = 0
        
        print("1) Susceptibility values in 3×3 grid:")
        print("\n   α →")
        print("β  ", end="")
        for alpha in alpha_values:
            print(f"   {alpha:.2f}  ", end="")
        print("\n↓  " + "-"*30)
        
        for beta in beta_values:
            print(f"{beta:.2f}", end="")
            for alpha in alpha_values:
                row = grid_data[(grid_data['beta'] == beta) & (grid_data['alpha'] == alpha)]
                if not row.empty:
                    chi = row.iloc[0]['chi_mean']
                    if (beta, alpha) == max_point:
                        print(f"  *{chi:.3f}", end="")
                    else:
                        print(f"   {chi:.3f}", end="")
                else:
                    print("   ----", end="")
            print()
        
        print(f"\n2) Maximum in grid:")
        print(f"   Location: (β={max_point[0]:.2f}, α={max_point[1]:.2f})")
        print(f"   χ_weight = {max_chi:.4f} ± {max_chi_err:.4f}")
        
        print(f"\n3) Comparison:")
        print(f"   Old peak (β={old_peak[0]}, α={old_peak[1]}): χ = {chi_old:.4f} ± {chi_old_err:.4f}")
        print(f"   New peak (β={max_point[0]:.2f}, α={max_point[1]:.2f}): χ = {max_chi:.4f} ± {max_chi_err:.4f}")
        
        if chi_old > 0:
            # Calculate percentage difference
            diff_percent = 100 * (max_chi - chi_old) / chi_old
            print(f"\n   Difference: {diff_percent:+.1f}%")
            
            if max_chi > chi_old:
                print(f"   → New region has HIGHER peak (by {abs(diff_percent):.1f}%)")
            else:
                print(f"   → Old location has HIGHER peak (by {abs(diff_percent):.1f}%)")
        
        # Check if FSS prediction is correct
        if max_point == (beta_center, alpha_center):
            print(f"\n   ✓ FSS prediction (β={beta_center}, α={alpha_center}) is at the center of the maximum!")
        else:
            dist = np.sqrt((max_point[0] - beta_center)**2 + (max_point[1] - alpha_center)**2)
            print(f"\n   → Maximum is at distance {dist:.3f} from FSS prediction")
        
        # Save detailed results
        print(f"\n{'='*60}")
        print(f"\nDetailed results saved to: {output_file}")
        
        # Also save summary
        summary = {
            'N': N,
            'num_steps': num_steps,
            'num_replicas': num_replicas,
            'grid_center': (beta_center, alpha_center),
            'grid_spacing': spacing,
            'old_peak': old_peak,
            'max_point': max_point,
            'max_chi': max_chi,
            'max_chi_err': max_chi_err,
            'old_chi': chi_old,
            'old_chi_err': chi_old_err,
            'elapsed_time': elapsed
        }
        
        import json
        summary_file = f'verification_summary_N{N}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Summary saved to: {summary_file}")
        
    except Exception as e:
        print(f"Error analyzing results: {e}")
        print(f"Please check the output file: {output_file}")

if __name__ == "__main__":
    run_verification_scan()