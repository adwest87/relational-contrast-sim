#!/usr/bin/env python3
"""
Monitor progress of the N=96 ridge scan.
Check for partial results and estimate completion time.
"""

import os
import glob
import time
from datetime import datetime, timedelta

def check_progress():
    """Check progress of ridge scan."""
    
    print("="*60)
    print("N=96 Ridge Scan Progress Monitor")
    print(f"Time: {datetime.now()}")
    print("="*60)
    
    # Check for output files
    possible_files = [
        'n96_ridge_results_*.csv',
        'results_n96_ridge*.csv',
        'fss_data/n96_ridge*.csv',
        'fss_data/results_n96*.csv'
    ]
    
    found_files = []
    for pattern in possible_files:
        files = glob.glob(pattern)
        found_files.extend(files)
    
    if not found_files:
        print("\nNo output files found yet.")
        print("The scan may still be running or saving to a different location.")
        
        # Check if process is running
        import subprocess
        try:
            result = subprocess.run(['pgrep', '-f', 'fss_narrow_scan'], 
                                  capture_output=True, text=True)
            if result.stdout.strip():
                print("\n✓ The scan process is still running (PID: {})".format(
                    result.stdout.strip().split('\n')[0]))
            else:
                print("\n✗ No scan process found running")
        except:
            pass
            
    else:
        print(f"\nFound {len(found_files)} output file(s):")
        
        for file in found_files:
            # Get file stats
            size = os.path.getsize(file) / 1024  # KB
            mtime = datetime.fromtimestamp(os.path.getmtime(file))
            
            print(f"\n  {file}:")
            print(f"    Size: {size:.1f} KB")
            print(f"    Last modified: {mtime}")
            
            # Try to count lines
            try:
                with open(file, 'r') as f:
                    lines = sum(1 for _ in f) - 1  # Subtract header
                print(f"    Data rows: {lines}")
                
                # Estimate progress
                total_points = 43
                replicas = 10
                expected_rows = total_points * replicas
                
                if lines > 0:
                    progress = (lines / expected_rows) * 100
                    print(f"    Progress: {lines}/{expected_rows} ({progress:.1f}%)")
                    
                    # Estimate completion time
                    elapsed = datetime.now() - mtime
                    if progress > 0:
                        total_time = elapsed * (100 / progress)
                        remaining = total_time - elapsed
                        eta = datetime.now() + remaining
                        print(f"    Estimated completion: {eta.strftime('%H:%M:%S')}")
                        
            except Exception as e:
                print(f"    Could not read file: {e}")
    
    # Check for log files
    print("\n" + "-"*60)
    log_files = glob.glob('*.log')
    if log_files:
        print(f"\nFound {len(log_files)} log file(s)")
        latest_log = max(log_files, key=os.path.getmtime)
        print(f"Latest: {latest_log}")
        
        # Show last few lines
        try:
            with open(latest_log, 'r') as f:
                lines = f.readlines()
                print("\nLast 5 lines:")
                for line in lines[-5:]:
                    print(f"  {line.rstrip()}")
        except:
            pass
    
    print("\n" + "="*60)
    print("\nTo run analysis when complete:")
    print("  python3 scripts/analyze_n96_ridge.py")

if __name__ == "__main__":
    check_progress()