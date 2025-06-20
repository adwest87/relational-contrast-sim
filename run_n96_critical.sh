#!/bin/bash
# Run N=96 at critical region only with proper equilibration

echo "Starting focused N=96 run at critical region"
echo "==========================================="

# Modify the fss_narrow_scan to use these parameters:
# - 400,000 total steps (2x the original)
# - 160,000 equilibration steps (3x the original)
# - 10 replicas for better statistics

cargo run --release --bin fss_narrow_scan -- \
  --pairs n96_critical_points.csv \
  --output fss_data/results_n96_critical.csv \
  --nodes 96 \
  --steps 400000 \
  --replicas 10 \
  --debug

echo "Completed critical region!"
