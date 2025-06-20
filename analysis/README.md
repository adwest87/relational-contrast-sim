# Analysis Results Organization

This directory contains all analysis results, figures, and scripts from the finite-size scaling analysis of the relational contrast system.

## Directory Structure

```
analysis/
├── figures/           # All generated figures
│   ├── fss/          # Finite-size scaling plots
│   ├── susceptibility/ # Susceptibility maps and heat maps
│   ├── ridge/        # Critical ridge analysis
│   ├── phase_diagram/ # Phase diagrams
│   └── paper/        # Publication-ready figures
├── scripts/          # Analysis scripts
│   ├── fss/          # FSS analysis scripts
│   ├── visualization/ # Figure generation scripts
│   ├── ridge/        # Ridge analysis scripts
│   └── data_processing/ # Data processing utilities
├── results/          # Numerical results and analysis outputs
└── data/            # Generated data files
```

## Key Results

### 1. Critical Point (Infinite Volume)
- **β∞ = 2.935 ± 0.003**
- **α∞ = 1.465 ± 0.001**

### 2. Critical Exponent
- **γ/ν = 2.035 ± 0.329** (consistent with mean-field/3D Ising)

### 3. Ridge Structure
- Critical ridge follows: **α ≈ 0.06β + 1.31**
- Peak positions for finite systems:
  - N=24: (β=2.900, α=1.490)
  - N=48: (β=2.910, α=1.480)
  - N=96: (β=2.930, α=1.470)

## Important Figures

### Finite-Size Scaling (`figures/fss/`)
- `enhanced_fss_analysis_*.png` - Complete FSS analysis with all panels
- `fss_susceptibility_scaling_*.png` - Log-log scaling of susceptibility
- `fss_critical_point_extrapolation_*.png` - β and α extrapolations

### Susceptibility Maps (`figures/susceptibility/`)
- `figure1_susceptibility_maps.png` - Main figure showing all three system sizes
- `figure1_updated_susceptibility_maps_*.png` - Version with N=96 ridge data

### Ridge Analysis (`figures/ridge/`)
- `critical_ridge_structure.png` - Comprehensive ridge analysis with 3D surface
- `ridge_alignment.png` - 2D ridge alignment across system sizes
- `n96_ridge_scan_visualization_*.png` - N=96 scan point distribution

## Key Scripts

### FSS Analysis
- `complete_fss_analysis_corrected.py` - Main FSS analysis using corrected peaks
- `enhanced_fss_analysis.py` - Robust analysis with error handling

### Visualization
- `create_updated_figure1.py` - Generate susceptibility maps
- `create_critical_ridge_figure.py` - Generate ridge structure figures
- `create_figure1_ridge_emphasis.py` - Enhanced visualization with ridge paths

### Ridge Analysis
- `analyze_n96_ridge.py` - Analyze ridge scan results
- `generate_n96_ridge_scan.py` - Generate adaptive ridge scan points

## Data Files

### Results (`results/`)
- `enhanced_fss_results_*.json` - Complete FSS analysis results
- `ridge_structure_analysis.txt` - Physical interpretation of ridge

### Data (`data/`)
- `n96_ridge_points.csv` - 43 points for refined N=96 scan
- Various verification point files

## Usage

To reproduce any analysis:

1. **FSS Analysis**: 
   ```bash
   python3 scripts/fss/enhanced_fss_analysis.py
   ```

2. **Generate Figures**:
   ```bash
   python3 scripts/visualization/create_updated_figure1.py
   python3 scripts/visualization/create_critical_ridge_figure.py
   ```

3. **Ridge Analysis**:
   ```bash
   python3 scripts/ridge/analyze_n96_ridge.py
   ```

## Summary

The analysis confirms:
1. Well-defined finite-size scaling with γ/ν ≈ 2.0
2. Systematic peak evolution along a critical ridge
3. Coupled order parameters requiring simultaneous tuning of β and α
4. Ridge structure indicating co-dimensional phase transition

All results support the emergence of geometric order at the critical point through the coupling of weight concentration and metric delocalization.