#!/usr/bin/env python3
"""Quick check of FSS test results"""

import pandas as pd

# Read test data
df = pd.read_csv('fss_data/test_n24.csv')

print("Test results for N=24:")
print(df[['beta', 'alpha', 'susceptibility', 'binder', 'mean_cos']].round(3))
print(f"\nMean acceptance: {df['acceptance'].mean():.3f}")
print(f"Mean autocorrelation time: {df['autocorr_time'].mean():.1f}")

# Find peak susceptibility in this small sample
peak_idx = df['susceptibility'].idxmax()
peak = df.loc[peak_idx]
print(f"\nPeak χ in test: {peak['susceptibility']:.2f} at β={peak['beta']}, α={peak['alpha']}")
print(f"Binder at peak: {peak['binder']:.3f}")