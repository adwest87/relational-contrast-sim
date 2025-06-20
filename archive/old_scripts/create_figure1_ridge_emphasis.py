#!/usr/bin/env python3
"""
Create Figure 1 variant that emphasizes the ridge structure
with ridge paths overlaid on the susceptibility maps.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from matplotlib.patches import Rectangle, FancyArrowPatch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from datetime import datetime

# Set publication quality defaults
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

def load_all_data():
    """Load data for all system sizes."""
    
    data_files = {
        24: ['fss_data/results_n24.csv'],
        48: ['fss_data/results_n48.csv'],
        96: ['fss_data/results_n96.csv', 'fss_data/results_n96_critical.csv']
    }
    
    all_data = {}
    
    for N, files in data_files.items():
        system_data = []
        for file in files:
            try:
                df = pd.read_csv(file)
                if 'n_nodes' in df.columns:
                    df = df[df['n_nodes'] == N]
                elif 'nodes' in df.columns:
                    df = df[df['nodes'] == N]
                system_data.append(df)
            except:
                continue
        
        if system_data:
            all_data[N] = pd.concat(system_data, ignore_index=True)
            # Average over replicas
            all_data[N] = all_data[N].groupby(['beta', 'alpha']).agg({
                'susceptibility': ['mean', 'std', 'count']
            }).reset_index()
            all_data[N].columns = ['beta', 'alpha', 'chi_mean', 'chi_std', 'count']
    
    return all_data

def extract_ridge_path(data, chi_threshold=0.8):
    """Extract ridge path from susceptibility data."""
    
    # Find points with high susceptibility
    max_chi = data['chi_mean'].max()
    ridge_points = data[data['chi_mean'] > chi_threshold * max_chi].copy()
    
    if len(ridge_points) < 3:
        return None
    
    # Sort by beta to create path
    ridge_points = ridge_points.sort_values('beta')
    
    # Smooth the path
    from scipy.interpolate import UnivariateSpline
    if len(ridge_points) > 3:
        spl = UnivariateSpline(ridge_points['beta'], ridge_points['alpha'], s=0.001)
        beta_smooth = np.linspace(ridge_points['beta'].min(), 
                                 ridge_points['beta'].max(), 50)
        alpha_smooth = spl(beta_smooth)
        
        return pd.DataFrame({'beta': beta_smooth, 'alpha': alpha_smooth})
    else:
        return ridge_points[['beta', 'alpha']]

def create_enhanced_susceptibility_map(ax, data, N, chi_min, chi_max, peak_loc, 
                                     all_peaks, ridge_path=None):
    """Create enhanced susceptibility map with ridge overlay."""
    
    # Create grid for interpolation
    beta_range = np.linspace(data['beta'].min(), data['beta'].max(), 60)
    alpha_range = np.linspace(data['alpha'].min(), data['alpha'].max(), 60)
    beta_grid, alpha_grid = np.meshgrid(beta_range, alpha_range)
    
    # Interpolate susceptibility
    points = data[['beta', 'alpha']].values
    values = data['chi_mean'].values
    chi_grid = griddata(points, values, (beta_grid, alpha_grid), method='cubic')
    
    # Smooth the grid slightly
    chi_grid = gaussian_filter(chi_grid, sigma=0.5)
    
    # Main contour plot
    levels = np.linspace(chi_min, chi_max, 25)
    cs = ax.contourf(beta_grid, alpha_grid, chi_grid, levels=levels, 
                     cmap='viridis', extend='both', alpha=0.9)
    
    # Add contour lines for better visibility
    contour_lines = ax.contour(beta_grid, alpha_grid, chi_grid, 
                              levels=8, colors='white', alpha=0.3, linewidths=1)
    
    # Plot ridge path if available
    if ridge_path is not None and len(ridge_path) > 2:
        ax.plot(ridge_path['beta'], ridge_path['alpha'], 
               'w--', linewidth=2.5, alpha=0.8, label='Ridge')
        ax.plot(ridge_path['beta'], ridge_path['alpha'], 
               'k--', linewidth=1.5, alpha=0.5)
    
    # Mark all peak locations
    for size, (b, a) in all_peaks.items():
        if size == N:
            # Current system peak - larger marker
            ax.scatter(b, a, s=250, marker='*', 
                      color='red', edgecolor='white', linewidth=2.5, zorder=20)
            ax.text(b+0.003, a+0.003, f'N={N}', 
                   fontsize=10, color='white', weight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.8))
        else:
            # Other system peaks - smaller markers
            marker = 's' if size < N else '^'
            ax.scatter(b, a, s=100, marker=marker, 
                      color='yellow', edgecolor='black', linewidth=1, 
                      alpha=0.7, zorder=15)
    
    # Title and labels
    ax.set_title(f'N = {N}', fontsize=14, weight='bold')
    ax.set_xlabel('β (weight coupling)')
    ax.set_ylabel('α (trace weight)')
    ax.grid(True, alpha=0.3, color='white', linewidth=0.5)
    
    # Add inset zoom
    axins = inset_axes(ax, width="45%", height="45%", loc='upper right',
                      bbox_to_anchor=(0.55, 0.55, 0.43, 0.43),
                      bbox_transform=ax.transAxes)
    
    # Define zoom region centered on peak
    zoom_size = 0.03
    beta_zoom = [peak_loc[0] - zoom_size, peak_loc[0] + zoom_size]
    alpha_zoom = [peak_loc[1] - zoom_size, peak_loc[1] + zoom_size]
    
    # Create zoom grid
    beta_zoom_grid = np.linspace(beta_zoom[0], beta_zoom[1], 40)
    alpha_zoom_grid = np.linspace(alpha_zoom[0], alpha_zoom[1], 40)
    beta_zgrid, alpha_zgrid = np.meshgrid(beta_zoom_grid, alpha_zoom_grid)
    
    # Interpolate in zoom region
    chi_zoom = griddata(points, values, (beta_zgrid, alpha_zgrid), method='cubic')
    chi_zoom = gaussian_filter(chi_zoom, sigma=0.5)
    
    # Plot zoom
    axins.contourf(beta_zgrid, alpha_zgrid, chi_zoom, levels=levels,
                  cmap='viridis', extend='both')
    
    # Mark peaks in zoom
    for size, (b, a) in all_peaks.items():
        if beta_zoom[0] <= b <= beta_zoom[1] and alpha_zoom[0] <= a <= alpha_zoom[1]:
            if size == N:
                axins.scatter(b, a, s=150, marker='*',
                             color='red', edgecolor='white', linewidth=2)
            else:
                marker = 's' if size < N else '^'
                axins.scatter(b, a, s=60, marker=marker,
                             color='yellow', edgecolor='black', linewidth=1)
    
    # Ridge in zoom
    if ridge_path is not None:
        zoom_ridge = ridge_path[(ridge_path['beta'] >= beta_zoom[0]) & 
                               (ridge_path['beta'] <= beta_zoom[1])]
        if len(zoom_ridge) > 1:
            axins.plot(zoom_ridge['beta'], zoom_ridge['alpha'], 
                      'w--', linewidth=2, alpha=0.8)
    
    axins.set_xlim(beta_zoom)
    axins.set_ylim(alpha_zoom)
    axins.tick_params(labelsize=8)
    axins.grid(True, alpha=0.3, color='white', linewidth=0.5)
    
    # Draw rectangle on main plot
    rect = Rectangle((beta_zoom[0], alpha_zoom[0]), 
                    beta_zoom[1]-beta_zoom[0], alpha_zoom[1]-alpha_zoom[0],
                    fill=False, edgecolor='red', linewidth=2, linestyle='-')
    ax.add_patch(rect)
    
    return cs

def create_figure1_ridge_emphasis(all_data):
    """Create Figure 1 with ridge emphasis."""
    
    # Peak locations
    peaks = {
        24: (2.900, 1.490),
        48: (2.910, 1.480),
        96: (2.930, 1.470)
    }
    
    # Get global chi range
    chi_values = []
    for data in all_data.values():
        chi_values.extend(data['chi_mean'].values)
    chi_min, chi_max = np.percentile(chi_values, [5, 95])
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    
    # Extract ridge paths
    ridge_paths = {}
    for N in [24, 48, 96]:
        if N in all_data:
            ridge_paths[N] = extract_ridge_path(all_data[N], chi_threshold=0.7)
    
    # Create maps
    for i, (N, ax) in enumerate(zip([24, 48, 96], axes)):
        if N in all_data:
            cs = create_enhanced_susceptibility_map(
                ax, all_data[N], N, chi_min, chi_max, 
                peaks[N], peaks, ridge_paths.get(N))
    
    # Add single colorbar
    cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
    cbar = plt.colorbar(cs, cax=cbar_ax)
    cbar.set_label('Susceptibility χ', fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    
    # Add flow arrows showing peak evolution
    for i in range(2):
        N1, N2 = [24, 48, 96][i], [24, 48, 96][i+1]
        p1, p2 = peaks[N1], peaks[N2]
        
        # Create curved arrow between panels
        ax1, ax2 = axes[i], axes[i+1]
        
        # Get axes positions in figure coordinates
        pos1 = ax1.get_position()
        pos2 = ax2.get_position()
        
        # Arrow from right edge of left panel to left edge of right panel
        arrow_start = (pos1.x1 + 0.01, pos1.y0 + pos1.height * 0.5)
        arrow_end = (pos2.x0 - 0.01, pos2.y0 + pos2.height * 0.5)
        
        arrow = FancyArrowPatch(arrow_start, arrow_end,
                               connectionstyle="arc3,rad=0.3",
                               arrowstyle='->', 
                               transform=fig.transFigure,
                               color='red', linewidth=3,
                               mutation_scale=25,
                               zorder=100)
        fig.add_artist(arrow)
        
        # Add text label
        mid_x = (arrow_start[0] + arrow_end[0]) / 2
        mid_y = (arrow_start[1] + arrow_end[1]) / 2 + 0.05
        fig.text(mid_x, mid_y, f'N={N1}→{N2}', 
                transform=fig.transFigure,
                ha='center', va='bottom',
                fontsize=10, color='red', weight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Overall title
    fig.suptitle('Susceptibility Ridge Evolution with System Size\n' + 
                'Peak Locations and High-χ Ridge Paths Shown',
                fontsize=16, y=0.98)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='w', linewidth=2, linestyle='--', 
               label='High-χ ridge (χ > 0.7 χ_max)'),
        Line2D([0], [0], marker='*', color='red', markersize=15,
               markeredgecolor='white', markeredgewidth=2, linestyle='',
               label='Peak (this N)'),
        Line2D([0], [0], marker='s', color='yellow', markersize=10,
               markeredgecolor='black', markeredgewidth=1, linestyle='',
               label='Peak (other N)')
    ]
    
    axes[1].legend(handles=legend_elements, loc='lower center', 
                  bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=True,
                  facecolor='white', edgecolor='black')
    
    # Adjust layout
    plt.subplots_adjust(left=0.05, right=0.92, wspace=0.20, bottom=0.15)
    
    # Save figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'figure1_ridge_emphasis_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\nSaved enhanced Figure 1: {filename}")
    
    return filename

def main():
    """Create Figure 1 with ridge emphasis."""
    
    print("="*60)
    print("Creating Figure 1 with Ridge Emphasis")
    print("="*60)
    
    # Load all data
    all_data = load_all_data()
    
    if not all_data:
        print("Error: No data loaded!")
        return
    
    print(f"\nLoaded data for N = {list(all_data.keys())}")
    
    # Create figure
    print("\nGenerating enhanced figure...")
    fig_file = create_figure1_ridge_emphasis(all_data)
    
    print("\n" + "="*60)
    print("Figure creation complete!")
    print(f"Enhanced figure: {fig_file}")
    print("="*60)

if __name__ == "__main__":
    main()