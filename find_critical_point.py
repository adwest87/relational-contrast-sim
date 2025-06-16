#!/usr/bin/env python3
"""Quick analysis to find critical point"""

import pandas as pd
import numpy as np

# Read the data
df = pd.read_csv('improved_narrow_results.csv')

# Sort by susceptibility to find peak
df_sorted = df.sort_values('susceptibility', ascending=False)

print("Top 10 points by susceptibility:")
print(df_sorted[['beta', 'alpha', 'susceptibility', 'mean_cos']].head(10))
print()

# Find peak
peak = df_sorted.iloc[0]
print(f"Peak susceptibility at β={peak['beta']:.2f}, α={peak['alpha']:.2f}, χ={peak['susceptibility']:.3f}")

# Check if we have enough resolution around the peak
beta_near = df[np.abs(df['beta'] - peak['beta']) <= 0.1]
alpha_near = df[np.abs(df['alpha'] - peak['alpha']) <= 0.1]

print(f"\nPoints near peak β: {len(beta_near)}")
print(f"Points near peak α: {len(alpha_near)}")

# Simple specific heat proxy from action variance
df['C_proxy'] = df['std_action']**2 / (48*47/2)  # Normalize by number of links

# Find specific heat peak
c_peak = df.loc[df['C_proxy'].idxmax()]
print(f"\nPeak specific heat proxy at β={c_peak['beta']:.2f}, α={c_peak['alpha']:.2f}, C={c_peak['C_proxy']:.1f}")

# Estimate critical region
high_chi = df[df['susceptibility'] > df['susceptibility'].mean() + df['susceptibility'].std()]
print(f"\nCritical region (high χ): β ∈ [{high_chi['beta'].min():.2f}, {high_chi['beta'].max():.2f}]")
print(f"                         α ∈ [{high_chi['alpha'].min():.2f}, {high_chi['alpha'].max():.2f}]")

# Save critical region for focused scan
critical_region = []
for beta in np.arange(2.9, 3.2, 0.01):
    for alpha in np.arange(1.4, 1.7, 0.01):
        critical_region.append(f"{beta:.2f},{alpha:.2f}")

with open('critical_region.csv', 'w') as f:
    f.write('\n'.join(critical_region))

print(f"\nGenerated {len(critical_region)} points in critical_region.csv for detailed scan")