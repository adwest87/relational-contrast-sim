#!/usr/bin/env python3
"""Generate minimal critical region points for N=96"""

# Based on N=24 and N=48 analysis, critical point is around (2.91, 1.48)
# Create a very focused grid

import numpy as np

# Very tight grid around critical point
beta_range = np.arange(2.88, 2.93, 0.01)  # 5 points
alpha_range = np.arange(1.46, 1.51, 0.01)  # 5 points

with open('n96_critical_points.csv', 'w') as f:
    for beta in beta_range:
        for alpha in alpha_range:
            f.write(f"{beta:.2f},{alpha:.2f}\n")

n_points = len(beta_range) * len(alpha_range)
print(f"Generated {n_points} critical points for N=96")
print(f"Beta range: {beta_range}")
print(f"Alpha range: {alpha_range}")

# Estimate runtime
time_per_point = 28 * 3600 / 132  # seconds (from your previous run)
total_time = n_points * time_per_point
print(f"\nEstimated runtime: {total_time/3600:.1f} hours ({total_time/3600/24:.1f} days)")
print(f"With proper equilibration: {total_time/3600 * 2:.1f} hours")

# Create run script
with open('run_n96_critical.sh', 'w') as f:
    f.write("""#!/bin/bash
# Run N=96 at critical region only with proper equilibration

echo "Starting focused N=96 run at critical region"
echo "==========================================="

# Modify the fss_narrow_scan to use these parameters:
# - 400,000 total steps (2x the original)
# - 160,000 equilibration steps (3x the original)
# - 10 replicas for better statistics

cargo run --release --bin fss_narrow_scan -- \\
  --pairs n96_critical_points.csv \\
  --output fss_data/results_n96_critical.csv \\
  --nodes 96 \\
  --steps 400000 \\
  --replicas 10 \\
  --debug

echo "Completed critical region!"
""")

print("\nCreated run_n96_critical.sh")
print("\nTo run: chmod +x run_n96_critical.sh && ./run_n96_critical.sh")
