#!/usr/bin/env python3
"""Monitor FSS runs in real-time"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import os

def check_file(filename):
    """Check if file exists and has data"""
    if not os.path.exists(filename):
        return None
    try:
        df = pd.read_csv(filename)
        if len(df) > 0:
            return df
    except:
        pass
    return None

# Files to monitor
files = {
    24: 'fss_data/results_n24.csv',
    48: 'fss_data/results_n48.csv', 
    96: 'fss_data/results_n96.csv'
}

# Critical point
beta_c = 2.90
alpha_c = 1.50

print("Monitoring FSS runs... (Ctrl+C to stop)")
print("-" * 60)

while True:
    # Check each file
    status = []
    data = {}
    
    for size, filename in files.items():
        df = check_file(filename)
        if df is None:
            status.append(f"N={size}: Not started")
        else:
            n_points = len(df)
            total_points = 121  # 11x11 grid
            pct = 100 * n_points / total_points
            
            # Find current peak
            if len(df) > 0:
                peak_idx = df['susceptibility'].idxmax()
                peak = df.loc[peak_idx]
                chi_max = peak['susceptibility']
                beta_peak = peak['beta']
                alpha_peak = peak['alpha']
                status.append(f"N={size}: {n_points}/{total_points} ({pct:.1f}%) - χ_max={chi_max:.2f} at ({beta_peak:.2f},{alpha_peak:.2f})")
            else:
                status.append(f"N={size}: {n_points}/{total_points} ({pct:.1f}%)")
            
            data[size] = df
    
    # Clear screen and print status
    os.system('clear' if os.name == 'posix' else 'cls')
    print("FSS Progress Monitor")
    print("=" * 60)
    print(f"Time: {time.strftime('%H:%M:%S')}")
    print(f"Critical point: β={beta_c}, α={alpha_c}")
    print("-" * 60)
    
    for s in status:
        print(s)
    
    # If we have data from multiple sizes, show scaling
    if len(data) >= 2:
        print("-" * 60)
        print("Preliminary scaling:")
        
        sizes = []
        chi_maxs = []
        
        for size, df in sorted(data.items()):
            sizes.append(size)
            chi_maxs.append(df['susceptibility'].max())
        
        if len(sizes) >= 2:
            # Simple log-log slope
            log_sizes = np.log(sizes)
            log_chi = np.log(chi_maxs)
            
            if len(sizes) == 2:
                slope = (log_chi[1] - log_chi[0]) / (log_sizes[1] - log_sizes[0])
            else:
                # Linear regression
                slope = np.polyfit(log_sizes, log_chi, 1)[0]
            
            print(f"γ/ν ≈ {slope:.3f} (from χ ~ L^(γ/ν))")
            print(f"Compare: 3D Ising γ/ν = 1.963")
    
    # Wait before next update
    time.sleep(10)