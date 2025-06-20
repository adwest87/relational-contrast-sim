#!/bin/bash
# Test universality along critical ridge for N=48
# Generated: 2025-06-20 21:48:17.527421

echo "=================================================="
echo "TESTING UNIVERSALITY ALONG CRITICAL RIDGE"
echo "=================================================="
echo ""
echo "System size: N=48"
echo "Ridge equation: α = 0.060β + 1.313"
echo "Testing 5 points along ridge"
echo ""

# Run parameters
STEPS=500000      # 5×10^5 MC steps
REPLICAS=20       # 20 replicas for good statistics
EQUILIBRATION=100000  # 20% for equilibration

echo "Parameters:"
echo "  MC steps: $STEPS"
echo "  Replicas: $REPLICAS"
echo "  Equilibration: $EQUILIBRATION"
echo ""

# Show points being tested
echo "Points to test:"
echo "  1. β=2.850, α=1.484"
echo "  2. β=2.875, α=1.486"
echo "  3. β=2.900, α=1.487"
echo "  4. β=2.925, α=1.488"
echo "  5. β=2.950, α=1.490"

echo ""
echo "Starting simulations..."
echo ""

# Run the scan
cargo run --release --bin fss_narrow_scan -- \
  --pairs ridge_universality_points.csv \
  --output ridge_universality_N48_$(date +%Y%m%d_%H%M%S).csv \
  --nodes 48 \
  --steps $STEPS \
  --replicas $REPLICAS \
  --debug

echo ""
echo "Simulations complete!"
echo ""
echo "To analyze results, run:"
echo "  python3 scripts/analyze_ridge_universality.py"
