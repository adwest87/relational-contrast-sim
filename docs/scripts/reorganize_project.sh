#!/bin/bash
# Project reorganization script for relational-contrast-sim
# This script reorganizes the source code into a cleaner directory structure

set -e  # Exit on error

echo "=== Relational Contrast Sim Project Reorganization ==="
echo "This script will reorganize the project structure."
echo "WARNING: This will move many files and update imports."
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 1
fi

# Create new directory structure
echo "Creating new directory structure..."
mkdir -p src/core
mkdir -p src/mc
mkdir -p src/analysis
mkdir -p src/platform
mkdir -p bin/core
mkdir -p bin/physics
mkdir -p bin/benchmarks
mkdir -p bin/experimental

# Move core graph implementations
echo "Moving core implementations..."
mv src/graph.rs src/core/ 2>/dev/null || true
mv src/graph_fast.rs src/core/ 2>/dev/null || true
mv src/graph_ultra_optimized.rs src/core/ 2>/dev/null || true

# Move Monte Carlo runners
echo "Moving Monte Carlo runners..."
mv src/fast_mc_integration.rs src/mc/ 2>/dev/null || true
mv src/importance_mc_integration.rs src/mc/ 2>/dev/null || true

# Move analysis modules
echo "Moving analysis modules..."
mv src/error_analysis.rs src/analysis/ 2>/dev/null || true
mv src/finite_size.rs src/analysis/ 2>/dev/null || true
mv src/importance_sampling.rs src/analysis/ 2>/dev/null || true
mv src/measure.rs src/analysis/ 2>/dev/null || true

# Move platform-specific implementations
echo "Moving platform-specific implementations..."
mv src/graph_m1_optimized.rs src/platform/ 2>/dev/null || true
mv src/graph_m1_accelerate.rs src/platform/ 2>/dev/null || true
mv src/graph_metal.rs src/platform/ 2>/dev/null || true

# Move binaries
echo "Moving binary files..."
# Core tools
mv src/bin/wide_scan.rs bin/core/ 2>/dev/null || true
mv src/bin/critical_finder.rs bin/core/ 2>/dev/null || true
mv src/bin/critical_finder_long.rs bin/core/ 2>/dev/null || true
mv src/bin/fss_analysis.rs bin/core/ 2>/dev/null || true
mv src/bin/multi_size_scan.rs bin/core/ 2>/dev/null || true

# Physics analysis
mv src/bin/physics_analysis/*.rs bin/physics/ 2>/dev/null || true
rmdir src/bin/physics_analysis 2>/dev/null || true

# Benchmarks
mv src/bin/benchmark_*.rs bin/benchmarks/ 2>/dev/null || true

# Experimental/other
mv src/bin/*.rs bin/experimental/ 2>/dev/null || true

# Create new lib.rs with updated module paths
echo "Creating updated lib.rs..."
cat > src/lib.rs << 'EOF'
// Core implementations
pub mod core {
    pub mod graph;
    pub mod graph_fast;
    pub mod graph_ultra_optimized;
}

// Monte Carlo integration
pub mod mc {
    pub mod fast_mc_integration;
    pub mod importance_mc_integration;
}

// Analysis tools
pub mod analysis {
    pub mod error_analysis;
    pub mod finite_size;
    pub mod importance_sampling;
    pub mod measure;
}

// Platform-specific implementations
#[cfg(target_arch = "aarch64")]
pub mod platform {
    pub mod graph_m1_optimized;
    
    #[cfg(target_os = "macos")]
    pub mod graph_m1_accelerate;
}

#[cfg(target_os = "macos")]
pub mod platform {
    pub mod graph_metal;
}

// Legacy modules (to be refactored)
pub mod projector;
pub mod observables;

// Re-exports for backward compatibility
pub use core::graph;
pub use core::graph_fast;
pub use core::graph_ultra_optimized;
pub use mc::fast_mc_integration;
pub use mc::importance_mc_integration;
pub use analysis::error_analysis;
pub use analysis::finite_size;
pub use analysis::importance_sampling;
pub use analysis::measure;

#[cfg(target_arch = "aarch64")]
pub use platform::graph_m1_optimized;

#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
pub use platform::graph_m1_accelerate;

#[cfg(target_os = "macos")]
pub use platform::graph_metal;
EOF

# Create a Python script to update imports in Rust files
echo "Creating import update script..."
cat > update_imports.py << 'EOF'
#!/usr/bin/env python3
import os
import re
import sys

# Mapping of old imports to new imports
import_mappings = {
    r'use (crate|scan)::graph::': r'use \1::core::graph::',
    r'use (crate|scan)::graph_fast::': r'use \1::core::graph_fast::',
    r'use (crate|scan)::graph_ultra_optimized::': r'use \1::core::graph_ultra_optimized::',
    r'use (crate|scan)::fast_mc_integration::': r'use \1::mc::fast_mc_integration::',
    r'use (crate|scan)::importance_mc_integration::': r'use \1::mc::importance_mc_integration::',
    r'use (crate|scan)::error_analysis::': r'use \1::analysis::error_analysis::',
    r'use (crate|scan)::finite_size::': r'use \1::analysis::finite_size::',
    r'use (crate|scan)::importance_sampling::': r'use \1::analysis::importance_sampling::',
    r'use (crate|scan)::measure::': r'use \1::analysis::measure::',
    r'use (crate|scan)::graph_m1_optimized::': r'use \1::platform::graph_m1_optimized::',
    r'use (crate|scan)::graph_m1_accelerate::': r'use \1::platform::graph_m1_accelerate::',
    r'use (crate|scan)::graph_metal::': r'use \1::platform::graph_metal::',
}

def update_imports_in_file(filepath):
    """Update imports in a single Rust file."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        original = content
        for old, new in import_mappings.items():
            content = re.sub(old, new, content)
        
        if content != original:
            with open(filepath, 'w') as f:
                f.write(content)
            print(f"Updated imports in {filepath}")
    except Exception as e:
        print(f"Error updating {filepath}: {e}")

def main():
    # Update all Rust files
    for root, dirs, files in os.walk('.'):
        # Skip target and .git directories
        if 'target' in root or '.git' in root:
            continue
        
        for file in files:
            if file.endswith('.rs'):
                filepath = os.path.join(root, file)
                update_imports_in_file(filepath)

if __name__ == "__main__":
    main()
EOF

# Update Cargo.toml to reflect new binary locations
echo "Creating Cargo.toml update script..."
cat > update_cargo_toml.py << 'EOF'
#!/usr/bin/env python3
import re

# Read Cargo.toml
with open('Cargo.toml', 'r') as f:
    content = f.read()

# Update binary paths
replacements = {
    'path = "src/bin/wide_scan.rs"': 'path = "bin/core/wide_scan.rs"',
    'path = "src/bin/critical_finder.rs"': 'path = "bin/core/critical_finder.rs"',
    'path = "src/bin/critical_finder_long.rs"': 'path = "bin/core/critical_finder_long.rs"',
    'path = "src/bin/fss_analysis.rs"': 'path = "bin/core/fss_analysis.rs"',
    'path = "src/bin/multi_size_scan.rs"': 'path = "bin/core/multi_size_scan.rs"',
    'path = "src/bin/benchmark_metal.rs"': 'path = "bin/benchmarks/benchmark_metal.rs"',
    'path = "src/bin/benchmark_m1.rs"': 'path = "bin/benchmarks/benchmark_m1.rs"',
    'path = "src/bin/benchmark_ultra_optimized.rs"': 'path = "bin/benchmarks/benchmark_ultra_optimized.rs"',
}

# Add patterns for physics analysis binaries
physics_binaries = [
    'unconventional_physics', 'quick_unconventional', 'low_temp_degeneracy',
    'publication_analysis', 'temperature_sweep', 'thermalization_test',
    'z3_order_parameter', 'critical_point_validation', 'finite_size_scaling',
    'phase_diagram_scan', 'broad_parameter_search'
]

for binary in physics_binaries:
    replacements[f'path = "src/bin/physics_analysis/{binary}.rs"'] = f'path = "bin/physics/{binary}.rs"'

# Update validation binaries
replacements['path = "src/bin/validation/quick_validation.rs"'] = 'path = "bin/experimental/quick_validation.rs"'

# Apply replacements
for old, new in replacements.items():
    content = content.replace(old, new)

# Handle remaining src/bin/ references
content = re.sub(r'path = "src/bin/([^"]+\.rs)"', r'path = "bin/experimental/\1"', content)

# Write updated Cargo.toml
with open('Cargo.toml', 'w') as f:
    f.write(content)

print("Updated Cargo.toml")
EOF

# Make scripts executable
chmod +x update_imports.py
chmod +x update_cargo_toml.py

# Run the update scripts
echo "Updating imports in Rust files..."
python3 update_imports.py

echo "Updating Cargo.toml..."
python3 update_cargo_toml.py

# Clean up
rm update_imports.py
rm update_cargo_toml.py

# Remove old empty directories
rmdir src/bin/validation 2>/dev/null || true
rmdir src/bin 2>/dev/null || true

echo ""
echo "=== Reorganization Complete ==="
echo ""
echo "New directory structure:"
echo "  src/"
echo "    ├── core/        - Core graph implementations"
echo "    ├── mc/          - Monte Carlo runners"
echo "    ├── analysis/    - Analysis utilities"
echo "    └── platform/    - Platform-specific code"
echo ""
echo "  bin/"
echo "    ├── core/        - Essential tools"
echo "    ├── physics/     - Physics analysis"
echo "    ├── benchmarks/  - Performance benchmarks"
echo "    └── experimental/ - Other experiments"
echo ""
echo "Next steps:"
echo "1. Run 'cargo check' to verify the build"
echo "2. Run 'cargo test' to ensure tests pass"
echo "3. Commit the changes to git"