#!/usr/bin/env python3
"""Generate a focused set of points for FSS around the critical point"""

import numpy as np

# Critical point from our analysis
beta_c = 2.90
alpha_c = 1.50

# For FSS, we want a smaller, focused grid
# Very close to critical point with fine spacing
beta_range = np.arange(2.85, 2.96, 0.01)
alpha_range = np.arange(1.45, 1.56, 0.01)

# Write the pairs
with open('fss_pairs.csv', 'w') as f:
    for beta in beta_range:
        for alpha in alpha_range:
            f.write(f"{beta:.2f},{alpha:.2f}\n")

n_points = len(beta_range) * len(alpha_range)
print(f"Generated {n_points} points in fss_pairs.csv")
print(f"Beta range: [{beta_range[0]:.2f}, {beta_range[-1]:.2f}]")
print(f"Alpha range: [{alpha_range[0]:.2f}, {alpha_range[-1]:.2f}]")

# Also create a minimal set for quick testing
test_points = [
    (2.88, 1.48),
    (2.89, 1.49),
    (2.90, 1.50),  # Critical point
    (2.91, 1.51),
    (2.92, 1.52),
]

with open('fss_test_pairs.csv', 'w') as f:
    for beta, alpha in test_points:
        f.write(f"{beta:.2f},{alpha:.2f}\n")

print(f"\nAlso created fss_test_pairs.csv with {len(test_points)} points for quick testing")