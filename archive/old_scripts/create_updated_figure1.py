#!/usr/bin/env python3
"""
Create updated Figure 1 showing susceptibility maps for N=24, 48, 96
with consistent color scales and peak locations marked.
Includes inset zooms of peak regions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib.patches import Rectangle
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
                # Filter for correct system size
                if 'n_nodes' in df.columns:
                    df = df[df['n_nodes'] == N]
                elif 'nodes' in df.columns:
                    df = df[df['nodes'] == N]
                system_data.append(df)
                print(f"Loaded {len(df)} points from {file}")
            except Exception as e:
                print(f"Could not load {file}: {e}")
        
        if system_data:
            all_data[N] = pd.concat(system_data, ignore_index=True)
            # Average over replicas
            all_data[N] = all_data[N].groupby(['beta', 'alpha']).agg({
                'susceptibility': ['mean', 'std', 'count']
            }).reset_index()
            all_data[N].columns = ['beta', 'alpha', 'chi_mean', 'chi_std', 'count']
            print(f"N={N}: {len(all_data[N])} unique (β,α) points")
    
    return all_data

def find_global_chi_range(all_data):
    """Find global min/max for consistent color scale."""
    
    chi_min = float('inf')
    chi_max = float('-inf')
    
    for N, data in all_data.items():
        chi_min = min(chi_min, data['chi_mean'].min())
        chi_max = max(chi_max, data['chi_mean'].max())
    
    print(f"\nGlobal χ range: [{chi_min:.1f}, {chi_max:.1f}]")
    return chi_min, chi_max

def create_susceptibility_map(ax, data, N, chi_min, chi_max, peak_loc, inset_pos='upper right'):
    """Create susceptibility map for one system size."""
    
    # Create grid for interpolation
    beta_range = np.linspace(data['beta'].min(), data['beta'].max(), 50)
    alpha_range = np.linspace(data['alpha'].min(), data['alpha'].max(), 50)
    beta_grid, alpha_grid = np.meshgrid(beta_range, alpha_range)
    
    # Interpolate susceptibility
    points = data[['beta', 'alpha']].values
    values = data['chi_mean'].values
    chi_grid = griddata(points, values, (beta_grid, alpha_grid), method='cubic')
    
    # Main contour plot
    levels = np.linspace(chi_min, chi_max, 20)
    cs = ax.contourf(beta_grid, alpha_grid, chi_grid, levels=levels, 
                     cmap='viridis', extend='both')
    
    # Add contour lines
    contour_lines = ax.contour(beta_grid, alpha_grid, chi_grid, 
                              levels=10, colors='black', alpha=0.2, linewidths=0.5)
    
    # Scatter actual data points
    scatter = ax.scatter(data['beta'], data['alpha'], 
                        c=data['chi_mean'], s=20, 
                        cmap='viridis', edgecolor='black', linewidth=0.5,
                        vmin=chi_min, vmax=chi_max, alpha=0.8)
    
    # Mark peak location
    ax.scatter(peak_loc[0], peak_loc[1], s=200, marker='*', 
              color='red', edgecolor='white', linewidth=2, zorder=10)
    ax.text(peak_loc[0]+0.005, peak_loc[1]+0.005, 'Peak', 
           fontsize=9, color='red', weight='bold',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    # Title and labels
    ax.set_title(f'N = {N}', fontsize=14, weight='bold')
    ax.set_xlabel('β (weight coupling)')
    ax.set_ylabel('α (trace weight)')
    ax.grid(True, alpha=0.3)
    
    # Add inset zoom of peak region
    if inset_pos == 'upper right':
        inset_loc = [0.55, 0.55, 0.42, 0.42]
    elif inset_pos == 'upper left':
        inset_loc = [0.03, 0.55, 0.42, 0.42]
    else:  # lower right
        inset_loc = [0.55, 0.03, 0.42, 0.42]
    
    axins = inset_axes(ax, width="40%", height="40%", 
                      bbox_to_anchor=inset_loc,
                      bbox_transform=ax.transAxes, loc=3)
    
    # Define zoom region
    zoom_size = 0.04
    beta_zoom = [peak_loc[0] - zoom_size, peak_loc[0] + zoom_size]
    alpha_zoom = [peak_loc[1] - zoom_size, peak_loc[1] + zoom_size]
    
    # Filter data for zoom region
    zoom_mask = ((data['beta'] >= beta_zoom[0]) & (data['beta'] <= beta_zoom[1]) &
                 (data['alpha'] >= alpha_zoom[0]) & (data['alpha'] <= alpha_zoom[1]))
    zoom_data = data[zoom_mask]
    
    if len(zoom_data) > 3:
        # Create finer grid for zoom
        beta_zoom_grid = np.linspace(beta_zoom[0], beta_zoom[1], 30)
        alpha_zoom_grid = np.linspace(alpha_zoom[0], alpha_zoom[1], 30)
        beta_zgrid, alpha_zgrid = np.meshgrid(beta_zoom_grid, alpha_zoom_grid)
        
        # Interpolate in zoom region
        chi_zoom = griddata(points, values, (beta_zgrid, alpha_zgrid), method='cubic')
        
        # Plot zoom
        axins.contourf(beta_zgrid, alpha_zgrid, chi_zoom, levels=levels,
                      cmap='viridis', extend='both')
        axins.scatter(zoom_data['beta'], zoom_data['alpha'], 
                     c=zoom_data['chi_mean'], s=40,
                     cmap='viridis', edgecolor='black', linewidth=0.5,
                     vmin=chi_min, vmax=chi_max)
        axins.scatter(peak_loc[0], peak_loc[1], s=100, marker='*',
                     color='red', edgecolor='white', linewidth=1.5)
    
    axins.set_xlim(beta_zoom)
    axins.set_ylim(alpha_zoom)
    axins.set_xticks([beta_zoom[0], peak_loc[0], beta_zoom[1]])
    axins.set_yticks([alpha_zoom[0], peak_loc[1], alpha_zoom[1]])
    axins.tick_params(labelsize=8)
    axins.grid(True, alpha=0.3)
    
    # Draw rectangle on main plot showing zoom region
    rect = Rectangle((beta_zoom[0], alpha_zoom[0]), 
                    beta_zoom[1]-beta_zoom[0], alpha_zoom[1]-alpha_zoom[0],
                    fill=False, edgecolor='red', linewidth=2, linestyle='--')
    ax.add_patch(rect)
    
    # Connect zoom to main plot
    from matplotlib.patches import ConnectionPatch
    con1 = ConnectionPatch(xyA=(beta_zoom[0], alpha_zoom[1]), xyB=(0, 1),
                          coordsA='data', coordsB='axes fraction',
                          axesA=ax, axesB=axins,
                          color='red', linewidth=1, linestyle='--')
    con2 = ConnectionPatch(xyA=(beta_zoom[1], alpha_zoom[1]), xyB=(1, 1),
                          coordsA='data', coordsB='axes fraction',
                          axesA=ax, axesB=axins,
                          color='red', linewidth=1, linestyle='--')
    ax.add_artist(con1)
    ax.add_artist(con2)
    
    return cs

def create_figure1(all_data):
    """Create the complete Figure 1."""
    
    # Peak locations (corrected)
    peaks = {
        24: (2.900, 1.490),
        48: (2.910, 1.480),
        96: (2.930, 1.470)
    }
    
    # Get global chi range
    chi_min, chi_max = find_global_chi_range(all_data)
    
    # Create figure with 3 panels
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Inset positions to avoid overlap
    inset_positions = ['upper right', 'upper left', 'lower right']
    
    # Create susceptibility maps
    for i, (N, ax) in enumerate(zip([24, 48, 96], axes)):
        if N in all_data:
            cs = create_susceptibility_map(ax, all_data[N], N, 
                                         chi_min, chi_max, peaks[N],
                                         inset_positions[i])
    
    # Add single colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = plt.colorbar(cs, cax=cbar_ax, label='Susceptibility χ')
    cbar.ax.tick_params(labelsize=10)
    
    # Add ridge evolution arrows between panels
    for i in range(2):
        ax1, ax2 = axes[i], axes[i+1]
        N1, N2 = [24, 48, 96][i], [24, 48, 96][i+1]
        
        # Transform peak coordinates to figure coordinates
        trans1 = ax1.transData + ax1.transAxes.inverted() + fig.transFigure.inverted()
        trans2 = ax2.transData + ax2.transAxes.inverted() + fig.transFigure.inverted()
        
        xy1 = trans1.transform(peaks[N1])
        xy2 = trans2.transform(peaks[N2])
        
        # Draw arrow
        arrow = plt.annotate('', xy=xy2, xytext=xy1,
                           xycoords='figure fraction',
                           arrowprops=dict(arrowstyle='->', 
                                         color='red', lw=2,
                                         connectionstyle="arc3,rad=0.3"))
        fig.add_artist(arrow)
    
    # Overall title
    fig.suptitle('Susceptibility Maps Showing Finite-Size Evolution Along Critical Ridge',
                fontsize=16, y=0.98)
    
    # Adjust layout
    plt.subplots_adjust(left=0.05, right=0.90, wspace=0.25)
    
    # Save figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'figure1_updated_susceptibility_maps_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\nSaved Figure 1: {filename}")
    
    # Also save without timestamp for easy reference
    plt.savefig('figure1_susceptibility_maps.png', dpi=300, bbox_inches='tight')
    print("Also saved as: figure1_susceptibility_maps.png")
    
    return filename

def create_summary_stats(all_data, peaks):
    """Create summary statistics table."""
    
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    for N in [24, 48, 96]:
        if N in all_data:
            data = all_data[N]
            peak = peaks[N]
            
            # Find data at peak
            peak_data = data[(data['beta'] == peak[0]) & (data['alpha'] == peak[1])]
            if len(peak_data) == 0:
                # Find nearest point
                distances = np.sqrt((data['beta'] - peak[0])**2 + 
                                  (data['alpha'] - peak[1])**2)
                nearest_idx = distances.idxmin()
                peak_data = data.iloc[[nearest_idx]]
            
            chi_at_peak = peak_data['chi_mean'].iloc[0]
            
            # Global maximum
            max_idx = data['chi_mean'].idxmax()
            max_chi = data.loc[max_idx, 'chi_mean']
            max_beta = data.loc[max_idx, 'beta']
            max_alpha = data.loc[max_idx, 'alpha']
            
            print(f"\nN = {N}:")
            print(f"  Total points: {len(data)}")
            print(f"  Peak location: (β={peak[0]:.3f}, α={peak[1]:.3f})")
            print(f"  χ at peak: {chi_at_peak:.2f}")
            print(f"  Global max: χ={max_chi:.2f} at (β={max_beta:.3f}, α={max_alpha:.3f})")
            print(f"  β range: [{data['beta'].min():.2f}, {data['beta'].max():.2f}]")
            print(f"  α range: [{data['alpha'].min():.2f}, {data['alpha'].max():.2f}]")

def main():
    """Create updated Figure 1."""
    
    print("="*60)
    print("Creating Updated Figure 1: Susceptibility Maps")
    print("="*60)
    
    # Load all data
    all_data = load_all_data()
    
    if not all_data:
        print("Error: No data loaded!")
        return
    
    # Peak locations
    peaks = {
        24: (2.900, 1.490),
        48: (2.910, 1.480),
        96: (2.930, 1.470)
    }
    
    # Create summary statistics
    create_summary_stats(all_data, peaks)
    
    # Create Figure 1
    print("\nGenerating figure...")
    fig_file = create_figure1(all_data)
    
    print("\n" + "="*60)
    print("Figure creation complete!")
    print(f"Main figure: {fig_file}")
    print("Reference copy: figure1_susceptibility_maps.png")
    print("="*60)

if __name__ == "__main__":
    main()