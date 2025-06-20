#!/usr/bin/env python3
"""
Quick verification scan for N=96 - reduced steps for faster results.
"""

import numpy as np
import pandas as pd
import subprocess
import time
from datetime import datetime
import os

def run_quick_verification():
    """Run quick verification scan for N=96 susceptibility peak."""
    
    # Parameters - reduced for quick scan
    N = 96
    num_steps = 20000    # Reduced from 100000
    num_replicas = 3     # Reduced from 5
    
    # Create 3×3 grid around (β=2.90, α=1.50)
    beta_center, alpha_center = 2.90, 1.50
    spacing = 0.01
    
    beta_values = [beta_center - spacing, beta_center, beta_center + spacing]
    alpha_values = [alpha_center - spacing, alpha_center, alpha_center + spacing]
    
    # Also check old peak location
    old_peak = (2.85, 1.55)
    
    print(f"=== QUICK N=96 Susceptibility Peak Verification ===")
    print(f"Time: {datetime.now()}")
    print(f"MC steps per point: {num_steps:,} (reduced for quick scan)")
    print(f"Replicas per point: {num_replicas}")
    print(f"\nScanning 3×3 grid around (β={beta_center}, α={alpha_center})")
    print(f"Also checking old peak at (β={old_peak[0]}, α={old_peak[1]})")
    print("\n" + "="*60 + "\n")
    
    # Create points file
    points_file = 'quick_verification_points.csv'
    with open(points_file, 'w') as f:
        # Grid points
        for beta in beta_values:
            for alpha in alpha_values:
                f.write(f"{beta:.2f},{alpha:.2f}\n")
        # Old peak
        f.write(f"{old_peak[0]:.2f},{old_peak[1]:.2f}\n")
    
    total_points = len(beta_values) * len(alpha_values) + 1
    print(f"Created {points_file} with {total_points} points")
    
    # Estimate runtime
    time_per_point = num_steps * num_replicas / 10000  # rough estimate: ~1s per 10k steps
    estimated_time = total_points * time_per_point
    print(f"Estimated runtime: {estimated_time:.0f} seconds ({estimated_time/60:.1f} minutes)")
    
    # Run simulation using Rust binary
    output_file = f'quick_verification_N{N}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    
    cmd = [
        'cargo', 'run', '--release', '--bin', 'fss_narrow_scan', '--',
        '--pairs', points_file,
        '--output', output_file,
        '--nodes', str(N),
        '--steps', str(num_steps),
        '--replicas', str(num_replicas)
    ]
    
    print(f"\nRunning simulation...")
    print("-" * 60)
    
    start_time = time.time()
    
    try:
        # Run the simulation with real-time output
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
        # Print output as it comes
        for line in iter(process.stdout.readline, ''):
            if line:
                print(line.rstrip())
        
        process.wait()
        
        if process.returncode != 0:
            print(f"Error: Process exited with code {process.returncode}")
            return
            
    except Exception as e:
        print(f"Error running simulation: {e}")
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
        grouped['chi_err'] = grouped['chi_std'] / np.sqrt(grouped['count'])
        
        # Sort by beta and alpha for consistent display
        grouped = grouped.sort_values(['beta', 'alpha'])
        
        # Print all results
        print("All susceptibility measurements:")
        print("-" * 50)
        for _, row in grouped.iterrows():
            beta, alpha = row['beta'], row['alpha']
            chi, err = row['chi_mean'], row['chi_err']
            
            if (beta, alpha) == old_peak:
                print(f"OLD PEAK: β={beta:.2f}, α={alpha:.2f}: χ = {chi:.4f} ± {err:.4f}")
            else:
                print(f"β={beta:.2f}, α={alpha:.2f}: χ = {chi:.4f} ± {err:.4f}")
        
        # Find maximum in grid (excluding old peak)
        grid_data = grouped[grouped.apply(lambda row: 
            (row['beta'], row['alpha']) != old_peak, axis=1)]
        
        if not grid_data.empty:
            max_idx = grid_data['chi_mean'].idxmax()
            max_beta = grid_data.loc[max_idx, 'beta']
            max_alpha = grid_data.loc[max_idx, 'alpha']
            max_chi = grid_data.loc[max_idx, 'chi_mean']
            max_err = grid_data.loc[max_idx, 'chi_err']
            
            print(f"\nMaximum in grid: (β={max_beta:.2f}, α={max_alpha:.2f})")
            print(f"χ_max = {max_chi:.4f} ± {max_err:.4f}")
            
            # Compare with old peak
            old_data = grouped[grouped.apply(lambda row: 
                (row['beta'], row['alpha']) == old_peak, axis=1)]
            
            if not old_data.empty:
                chi_old = old_data.iloc[0]['chi_mean']
                err_old = old_data.iloc[0]['chi_err']
                
                diff_percent = 100 * (max_chi - chi_old) / chi_old
                print(f"\nOld peak χ = {chi_old:.4f} ± {err_old:.4f}")
                print(f"Difference: {diff_percent:+.1f}%")
                
                if max_chi > chi_old:
                    print(f"→ New region has HIGHER susceptibility")
                else:
                    print(f"→ Old location has HIGHER susceptibility")
            
            # Check distance from FSS prediction
            dist = np.sqrt((max_beta - beta_center)**2 + (max_alpha - alpha_center)**2)
            print(f"\nDistance from FSS prediction (β={beta_center}, α={alpha_center}): {dist:.3f}")
            
            if dist < 0.001:
                print("✓ FSS prediction is at the maximum!")
            elif dist <= spacing:
                print("✓ FSS prediction is very close to maximum")
            else:
                print(f"→ Maximum is {dist/spacing:.1f} grid spacings away")
        
        print(f"\n{'='*60}")
        print(f"\nDetailed results saved to: {output_file}")
        print("\nNOTE: This was a quick scan with reduced statistics.")
        print("For publication-quality results, run with more steps and replicas.")
        
    except Exception as e:
        print(f"Error analyzing results: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_quick_verification()