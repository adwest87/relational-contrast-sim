#!/bin/bash
# Refined N=96 ridge scan with adaptive sampling
# Generated: 2025-06-20 19:42:33.861310
# Total points: 43

echo "Starting refined N=96 ridge scan"
echo "================================"
echo "Total points: 43"
echo "MC steps per point: 300,000"
echo "Focus on true critical region near (β=2.93, α=1.47)"
echo ""

# Run the scan with enhanced equilibration
cargo run --release --bin fss_narrow_scan -- \
  --pairs n96_ridge_points.csv \
  --output n96_ridge_results_20250620_194233.csv \
  --nodes 96 \
  --steps 300000 \
  --replicas 10 \
  --debug

echo ""
echo "Scan completed!"
echo "Run 'python3 scripts/analyze_n96_ridge.py' to analyze results"
