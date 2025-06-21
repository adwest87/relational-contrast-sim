#!/bin/bash
# Script to tidy up loose files in the relational-contrast-sim project

set -e

echo "=== Tidying up loose files ==="
echo ""

# Create directories for organization
mkdir -p docs/scripts
mkdir -p docs/patches
mkdir -p archive/logs
mkdir -p archive/benchmarks
mkdir -p src/utils

# Move patch files to docs/patches
echo "Moving patch files..."
mv src/incremental_triangle_update.patch docs/patches/ 2>/dev/null || true
mv benchmarks/triangle_optimization.patch docs/patches/ 2>/dev/null || true

# Move the reorganization script to docs/scripts
echo "Moving maintenance scripts..."
mv reorganize_project.sh docs/scripts/ 2>/dev/null || true

# Move cleanup summary to docs
echo "Moving documentation..."
mv CLEANUP_SUMMARY.md docs/ 2>/dev/null || true

# Archive old log files
echo "Archiving log files..."
find . -name "*.log" -not -path "./target/*" -not -path "./.git/*" -exec mv {} archive/logs/ \; 2>/dev/null || true

# Move benchmark files to appropriate location
echo "Organizing benchmark files..."
mv benchmarks/test_benchmark.sh archive/benchmarks/ 2>/dev/null || true
mv benchmarks/test_gpu_batched.rs src/bin/benchmark_gpu_batched.rs 2>/dev/null || true
rmdir benchmarks 2>/dev/null || true

# Move utility modules in src/ to utils subdirectory
echo "Organizing src/ utility modules..."
mv src/config.rs src/utils/ 2>/dev/null || true
mv src/output.rs src/utils/ 2>/dev/null || true
mv src/ridge.rs src/utils/ 2>/dev/null || true
mv src/rng.rs src/utils/ 2>/dev/null || true

# Move benchmark file
mv src/benchmark_cache_optimization.rs src/bin/ 2>/dev/null || true

# Update lib.rs to include utils module
echo "Updating lib.rs for utils module..."
cat >> src/lib.rs << 'EOF'

// Utility modules
pub mod utils {
    pub mod config;
    pub mod output;
    pub mod ridge;
    pub mod rng;
}

// Re-exports for backward compatibility
pub use utils::config;
pub use utils::output;
pub use utils::ridge;
pub use utils::rng;
EOF

# Clean up empty directories
find . -type d -empty -not -path "./target/*" -not -path "./.git/*" -delete 2>/dev/null || true

echo ""
echo "=== File Organization Complete ==="
echo ""
echo "Changes made:"
echo "  - Patch files moved to docs/patches/"
echo "  - Scripts moved to docs/scripts/"
echo "  - Log files archived to archive/logs/"
echo "  - Utility modules moved to src/utils/"
echo "  - Benchmark files organized"
echo ""
echo "Root directory now contains only:"
ls -la | grep -E "^-" | grep -v "^\." | awk '{print "  - " $NF}'
echo ""
echo "Next steps:"
echo "1. Update imports if using moved utility modules"
echo "2. Run 'cargo check' to verify build"
echo "3. The reorganize_project.sh script is now in docs/scripts/"