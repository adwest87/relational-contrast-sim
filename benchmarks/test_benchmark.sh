#!/bin/bash

echo "Testing benchmark compilation..."

# First, build the library
echo "Building library..."
cargo build --release 2>&1 | tail -20

# Then try to build the benchmark
echo -e "\nBuilding benchmark..."
cargo build --release --bin benchmark_comparison 2>&1 | tail -20

# If successful, run it
if [ $? -eq 0 ]; then
    echo -e "\nRunning benchmark..."
    cargo run --release --bin benchmark_comparison
else
    echo -e "\nBuild failed. Check errors above."
fi