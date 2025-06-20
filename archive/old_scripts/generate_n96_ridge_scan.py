#!/usr/bin/env python3
"""
Generate refined N=96 ridge scan with adaptive sampling.
Focus on the true critical region with higher density near the observed maximum.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

def generate_ridge_points():
    """Generate adaptive ridge points for N=96 scan."""
    
    # Known peak locations
    peak_n96 = (2.93, 1.47)  # Observed maximum
    peak_n48 = (2.91, 1.48)  # From previous analysis
    peak_n24 = (2.90, 1.49)  # From previous analysis
    
    # Ridge endpoints
    start = (2.88, 1.46)
    end = (2.94, 1.52)
    
    points = []
    
    # 1. Generate 25 points in high-density region (±0.02 around peak)
    print("Generating high-density points around peak...")
    beta_peak, alpha_peak = peak_n96
    
    # Create a grid with adaptive spacing
    n_dense = 25
    # Use finer spacing near center
    beta_offsets = []
    alpha_offsets = []
    
    for i in range(5):
        for j in range(5):
            # Distance from center (0-2 in each direction)
            dist_i = abs(i - 2)
            dist_j = abs(j - 2)
            
            # Adaptive spacing: finer near center
            spacing = 0.005 if (dist_i <= 1 and dist_j <= 1) else 0.01
            
            beta_off = (i - 2) * spacing
            alpha_off = (j - 2) * spacing
            
            beta = beta_peak + beta_off
            alpha = alpha_peak + alpha_off
            
            # Keep within ±0.02 range
            if abs(beta_off) <= 0.02 and abs(alpha_off) <= 0.02:
                points.append((beta, alpha))
    
    # Ensure we have exactly 25 points
    points = points[:n_dense]
    
    # 2. Generate 15 points along the broader ridge
    print("Generating ridge points...")
    
    # Define ridge path (approximately linear between endpoints)
    n_ridge = 15
    
    # Exclude the dense region
    ridge_points = []
    for t in np.linspace(0, 1, n_ridge + 10):
        beta = start[0] + t * (end[0] - start[0])
        alpha = start[1] + t * (end[1] - start[1])
        
        # Skip if too close to peak region
        if abs(beta - beta_peak) > 0.025 or abs(alpha - alpha_peak) > 0.025:
            ridge_points.append((beta, alpha))
    
    # Take 15 points evenly distributed
    if len(ridge_points) > n_ridge:
        indices = np.linspace(0, len(ridge_points)-1, n_ridge, dtype=int)
        ridge_points = [ridge_points[i] for i in indices]
    
    points.extend(ridge_points)
    
    # 3. Add 5 checkpoint points
    print("Adding checkpoint points...")
    checkpoints = [
        peak_n24,  # N=24 peak
        peak_n48,  # N=48 peak
        (2.85, 1.55),  # Old N=96 location
        (2.90, 1.50),  # FSS prediction
        peak_n96   # Observed N=96 maximum (ensure it's included)
    ]
    
    # Remove duplicates while preserving order
    all_points = []
    seen = set()
    
    for p in points + checkpoints:
        # Round to avoid floating point issues
        p_round = (round(p[0], 3), round(p[1], 3))
        if p_round not in seen:
            seen.add(p_round)
            all_points.append(p)
    
    print(f"\nTotal unique points: {len(all_points)}")
    
    # Sort by beta then alpha for clarity
    all_points.sort(key=lambda x: (x[0], x[1]))
    
    return all_points, peak_n24, peak_n48, peak_n96

def visualize_ridge_scan(points, peak_n24, peak_n48, peak_n96):
    """Create visualization of the ridge scan points."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Convert points to arrays
    points_arr = np.array(points)
    betas = points_arr[:, 0]
    alphas = points_arr[:, 1]
    
    # Plot 1: All points with peaks marked
    ax1.scatter(betas, alphas, c='blue', alpha=0.5, s=30, label='Scan points')
    
    # Mark special points
    ax1.scatter(*peak_n24, c='green', s=200, marker='s', label='N=24 peak', edgecolor='black')
    ax1.scatter(*peak_n48, c='orange', s=200, marker='^', label='N=48 peak', edgecolor='black')
    ax1.scatter(*peak_n96, c='red', s=200, marker='*', label='N=96 maximum', edgecolor='black')
    ax1.scatter(2.90, 1.50, c='purple', s=150, marker='D', label='FSS prediction', edgecolor='black')
    ax1.scatter(2.85, 1.55, c='gray', s=150, marker='X', label='Old N=96 location', edgecolor='black')
    
    # Draw ridge line
    ridge_start = (2.88, 1.46)
    ridge_end = (2.94, 1.52)
    ax1.plot([ridge_start[0], ridge_end[0]], [ridge_start[1], ridge_end[1]], 
             'k--', alpha=0.3, label='Ridge path')
    
    # Highlight dense region
    circle = plt.Circle(peak_n96, 0.02, fill=False, edgecolor='red', 
                       linestyle='--', linewidth=2, label='Dense region')
    ax1.add_patch(circle)
    
    ax1.set_xlabel('β (weight coupling)', fontsize=12)
    ax1.set_ylabel('α (trace weight)', fontsize=12)
    ax1.set_title('N=96 Ridge Scan Points', fontsize=14)
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Density heatmap
    # Create 2D histogram to show point density
    H, xedges, yedges = np.histogram2d(betas, alphas, bins=20)
    
    im = ax2.imshow(H.T, origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                    aspect='auto', cmap='YlOrRd', interpolation='gaussian')
    
    # Overlay peak locations
    ax2.scatter(*peak_n24, c='green', s=200, marker='s', edgecolor='black')
    ax2.scatter(*peak_n48, c='orange', s=200, marker='^', edgecolor='black')
    ax2.scatter(*peak_n96, c='red', s=200, marker='*', edgecolor='black')
    
    # Draw arrow showing system size evolution
    ax2.annotate('', xy=peak_n48, xytext=peak_n24,
                arrowprops=dict(arrowstyle='->', color='green', lw=2))
    ax2.annotate('', xy=peak_n96, xytext=peak_n48,
                arrowprops=dict(arrowstyle='->', color='orange', lw=2))
    
    ax2.text(peak_n24[0]-0.002, peak_n24[1]+0.003, 'N=24', fontsize=10, ha='right')
    ax2.text(peak_n48[0]-0.002, peak_n48[1]+0.003, 'N=48', fontsize=10, ha='right')
    ax2.text(peak_n96[0]+0.002, peak_n96[1]-0.003, 'N=96', fontsize=10, ha='left')
    
    ax2.set_xlabel('β (weight coupling)', fontsize=12)
    ax2.set_ylabel('α (trace weight)', fontsize=12)
    ax2.set_title('Point Density & Peak Evolution', fontsize=14)
    
    plt.colorbar(im, ax=ax2, label='Point density')
    
    plt.tight_layout()
    
    # Save figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'n96_ridge_scan_visualization_{timestamp}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {filename}")
    
    return filename

def create_run_script(points):
    """Create shell script to run the scan."""
    
    script_content = f"""#!/bin/bash
# Refined N=96 ridge scan with adaptive sampling
# Generated: {datetime.now()}
# Total points: {len(points)}

echo "Starting refined N=96 ridge scan"
echo "================================"
echo "Total points: {len(points)}"
echo "MC steps per point: 300,000"
echo "Focus on true critical region near (β=2.93, α=1.47)"
echo ""

# Run the scan with enhanced equilibration
cargo run --release --bin fss_narrow_scan -- \\
  --pairs n96_ridge_points.csv \\
  --output n96_ridge_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv \\
  --nodes 96 \\
  --steps 300000 \\
  --replicas 10 \\
  --debug

echo ""
echo "Scan completed!"
echo "Run 'python3 scripts/analyze_n96_ridge.py' to analyze results"
"""
    
    with open('run_n96_ridge_scan.sh', 'w') as f:
        f.write(script_content)
    
    print("Created run script: run_n96_ridge_scan.sh")

def main():
    """Generate ridge scan points and visualization."""
    
    print("="*60)
    print("N=96 Refined Ridge Scan Generator")
    print("="*60)
    
    # Generate points
    points, peak_n24, peak_n48, peak_n96 = generate_ridge_points()
    
    # Save points to CSV
    with open('n96_ridge_points.csv', 'w') as f:
        for beta, alpha in points:
            f.write(f"{beta:.3f},{alpha:.3f}\n")
    
    print(f"\nSaved {len(points)} points to: n96_ridge_points.csv")
    
    # Create visualization
    print("\nCreating visualization...")
    vis_file = visualize_ridge_scan(points, peak_n24, peak_n48, peak_n96)
    
    # Create run script
    create_run_script(points)
    
    # Summary statistics
    points_arr = np.array(points)
    print("\n" + "="*60)
    print("SUMMARY:")
    print(f"Total points: {len(points)}")
    print(f"β range: [{points_arr[:, 0].min():.3f}, {points_arr[:, 0].max():.3f}]")
    print(f"α range: [{points_arr[:, 1].min():.3f}, {points_arr[:, 1].max():.3f}]")
    
    # Count points near peak
    peak_beta, peak_alpha = peak_n96
    near_peak = sum(1 for b, a in points 
                   if abs(b - peak_beta) <= 0.02 and abs(a - peak_alpha) <= 0.02)
    print(f"Points in dense region (±0.02 of peak): {near_peak}")
    
    print("\nKey locations included:")
    print(f"  - N=96 maximum: β={peak_n96[0]:.3f}, α={peak_n96[1]:.3f}")
    print(f"  - N=48 peak: β={peak_n48[0]:.3f}, α={peak_n48[1]:.3f}")
    print(f"  - N=24 peak: β={peak_n24[0]:.3f}, α={peak_n24[1]:.3f}")
    print(f"  - FSS prediction: β=2.900, α=1.500")
    print(f"  - Old location: β=2.850, α=1.550")
    
    # Estimate runtime
    time_per_point = 300000 * 10 / 50000  # rough estimate
    total_time = len(points) * time_per_point
    print(f"\nEstimated runtime: {total_time/60:.1f} minutes ({total_time/3600:.1f} hours)")
    
    print("\nTo run the scan:")
    print("  chmod +x run_n96_ridge_scan.sh")
    print("  ./run_n96_ridge_scan.sh")

if __name__ == "__main__":
    main()