#!/usr/bin/env python3
"""
Analyze existing N=96 data to verify susceptibility peak location.
Checks if the true peak is near (β=2.90, α=1.50) as FSS analysis suggests.
"""

import numpy as np
import pandas as pd
from datetime import datetime

def analyze_verification():
    """Analyze existing N=96 data for peak verification."""
    
    print(f"=== N=96 Susceptibility Peak Verification Analysis ===")
    print(f"Time: {datetime.now()}")
    print("="*60 + "\n")
    
    # FSS prediction and old peak
    beta_fss, alpha_fss = 2.90, 1.50
    beta_old, alpha_old = 2.85, 1.55
    
    # Define search regions
    spacing = 0.01
    beta_range = [beta_fss - spacing, beta_fss, beta_fss + spacing]
    alpha_range = [alpha_fss - spacing, alpha_fss, alpha_fss + spacing]
    
    # Load existing data
    data_files = [
        'fss_data/results_n96.csv',
        'fss_data/results_n96_critical.csv'
    ]
    
    all_data = []
    for file in data_files:
        try:
            df = pd.read_csv(file)
            if 'nodes' in df.columns:
                df = df[df['nodes'] == 96]
            all_data.append(df)
            print(f"Loaded {len(df)} entries from {file}")
        except Exception as e:
            print(f"Could not load {file}: {e}")
    
    if not all_data:
        print("No data files found!")
        return
    
    # Combine all data
    df = pd.concat(all_data, ignore_index=True)
    print(f"\nTotal N=96 data points: {len(df)}")
    
    # Group by (beta, alpha) and average
    grouped = df.groupby(['beta', 'alpha']).agg({
        'susceptibility': ['mean', 'std', 'count']
    }).reset_index()
    
    grouped.columns = ['beta', 'alpha', 'chi_mean', 'chi_std', 'count']
    grouped['chi_err'] = grouped['chi_std'] / np.sqrt(grouped['count'])
    
    print(f"Unique parameter points: {len(grouped)}")
    
    # Find data in the verification region
    grid_data = grouped[
        (grouped['beta'] >= beta_range[0]) & 
        (grouped['beta'] <= beta_range[-1]) &
        (grouped['alpha'] >= alpha_range[0]) & 
        (grouped['alpha'] <= alpha_range[-1])
    ]
    
    print(f"\nPoints in verification grid around (β={beta_fss}, α={alpha_fss}): {len(grid_data)}")
    
    if len(grid_data) == 0:
        print("No data found in the verification region!")
        print("\nSearching for nearest points...")
        
        # Find nearest points
        grouped['dist'] = np.sqrt(
            (grouped['beta'] - beta_fss)**2 + 
            (grouped['alpha'] - alpha_fss)**2
        )
        nearest = grouped.nsmallest(5, 'dist')
        
        print("\nNearest 5 points to FSS prediction:")
        for _, row in nearest.iterrows():
            print(f"  β={row['beta']:.2f}, α={row['alpha']:.2f}: "
                  f"χ = {row['chi_mean']:.4f} ± {row['chi_err']:.4f} "
                  f"(dist = {row['dist']:.3f})")
    else:
        # Display grid data
        print("\n1) Susceptibility values in verification region:")
        print("-" * 50)
        
        # Create grid display
        print("\n   α →")
        print("β  ", end="")
        for alpha in sorted(grid_data['alpha'].unique()):
            print(f"   {alpha:.2f}  ", end="")
        print("\n↓  " + "-"*40)
        
        for beta in sorted(grid_data['beta'].unique()):
            print(f"{beta:.2f}", end="")
            for alpha in sorted(grid_data['alpha'].unique()):
                row = grid_data[(grid_data['beta'] == beta) & (grid_data['alpha'] == alpha)]
                if not row.empty:
                    chi = row.iloc[0]['chi_mean']
                    print(f"   {chi:.3f}", end="")
                else:
                    print("   ----", end="")
            print()
        
        # Find maximum
        max_idx = grid_data['chi_mean'].idxmax()
        max_beta = grid_data.loc[max_idx, 'beta']
        max_alpha = grid_data.loc[max_idx, 'alpha']
        max_chi = grid_data.loc[max_idx, 'chi_mean']
        max_err = grid_data.loc[max_idx, 'chi_err']
        max_count = grid_data.loc[max_idx, 'count']
        
        print(f"\n2) Maximum in verification region:")
        print(f"   Location: (β={max_beta:.2f}, α={max_alpha:.2f})")
        print(f"   χ_weight = {max_chi:.4f} ± {max_err:.4f}")
        print(f"   Based on {max_count:.0f} measurements")
    
    # Check old peak location
    old_data = grouped[
        (grouped['beta'] == beta_old) & 
        (grouped['alpha'] == alpha_old)
    ]
    
    if not old_data.empty:
        chi_old = old_data.iloc[0]['chi_mean']
        err_old = old_data.iloc[0]['chi_err']
        count_old = old_data.iloc[0]['count']
        
        print(f"\n3) Old peak location (β={beta_old}, α={alpha_old}):")
        print(f"   χ_weight = {chi_old:.4f} ± {err_old:.4f}")
        print(f"   Based on {count_old:.0f} measurements")
        
        if len(grid_data) > 0:
            diff_percent = 100 * (max_chi - chi_old) / chi_old
            print(f"\n   Difference: {diff_percent:+.1f}%")
            
            if max_chi > chi_old:
                print(f"   → Verification region has HIGHER peak")
            else:
                print(f"   → Old location has HIGHER peak")
            
            # Check distance from FSS prediction
            dist = np.sqrt((max_beta - beta_fss)**2 + (max_alpha - alpha_fss)**2)
            print(f"\n   Distance from FSS prediction: {dist:.3f}")
            
            if dist < 0.001:
                print("   ✓ FSS prediction is exactly at the maximum!")
            elif dist <= spacing:
                print("   ✓ FSS prediction is very close to maximum")
            else:
                print(f"   → Maximum is {dist/spacing:.1f} grid spacings away")
    else:
        print(f"\nNo data found at old peak location (β={beta_old}, α={alpha_old})")
    
    # Find global maximum
    print("\n" + "="*60)
    print("\n4) Global maximum in all N=96 data:")
    
    global_max_idx = grouped['chi_mean'].idxmax()
    global_max = grouped.loc[global_max_idx]
    
    print(f"   Location: (β={global_max['beta']:.2f}, α={global_max['alpha']:.2f})")
    print(f"   χ_weight = {global_max['chi_mean']:.4f} ± {global_max['chi_err']:.4f}")
    print(f"   Based on {global_max['count']:.0f} measurements")
    
    dist_from_fss = np.sqrt(
        (global_max['beta'] - beta_fss)**2 + 
        (global_max['alpha'] - alpha_fss)**2
    )
    print(f"   Distance from FSS prediction: {dist_from_fss:.3f}")
    
    # Summary
    print("\n" + "="*60)
    print("\nSUMMARY:")
    
    if len(grid_data) > 0:
        print(f"✓ Found {len(grid_data)} points in verification region")
        print(f"✓ Region maximum at (β={max_beta:.2f}, α={max_alpha:.2f})")
        
        if dist < spacing:
            print("✓ FSS prediction is confirmed - very close to local maximum")
        else:
            print(f"! FSS prediction is {dist/spacing:.1f} grid spacings from local maximum")
    else:
        print("! No data in verification region - need to run focused scan")
        print(f"  Suggested: 3×3 grid around (β={beta_fss}, α={alpha_fss}) with spacing {spacing}")
    
    print(f"\nGlobal maximum suggests critical point near: "
          f"(β={global_max['beta']:.3f}, α={global_max['alpha']:.3f})")

if __name__ == "__main__":
    analyze_verification()