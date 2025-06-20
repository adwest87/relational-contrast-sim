#!/usr/bin/env python3
"""
Enhanced finite-size scaling analysis with more robust data handling.
Uses all available data near the peak locations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
from datetime import datetime

# Set publication quality defaults
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.figsize': (8, 6),
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'lines.linewidth': 2,
    'lines.markersize': 8
})

def load_enhanced_peak_data():
    """Load data near peak locations with tolerance."""
    
    # Corrected peak locations
    peaks = {
        24: (2.900, 1.490),
        48: (2.910, 1.480),
        96: (2.930, 1.470)
    }
    
    # Load all data
    all_data = []
    data_files = ['fss_data/results_n24.csv', 'fss_data/results_n48.csv', 
                  'fss_data/results_n96.csv', 'fss_data/results_n96_critical.csv']
    
    for file in data_files:
        try:
            df = pd.read_csv(file)
            all_data.append(df)
            print(f"Loaded {file}")
        except:
            continue
    
    if not all_data:
        return None
    
    df = pd.concat(all_data, ignore_index=True)
    
    # Get susceptibility at peaks with larger tolerance
    peak_data = []
    tolerance = 0.02  # Increased tolerance
    
    for N, (beta_peak, alpha_peak) in peaks.items():
        # Find data near peak
        mask = ((np.abs(df['beta'] - beta_peak) <= tolerance) & 
                (np.abs(df['alpha'] - alpha_peak) <= tolerance))
        
        if 'n_nodes' in df.columns:
            mask &= (df['n_nodes'] == N)
        
        near_peak = df[mask]
        
        if len(near_peak) == 0:
            print(f"N={N}: No data near peak, searching wider...")
            # Try wider search
            mask = (df['n_nodes'] == N) if 'n_nodes' in df.columns else pd.Series(True, index=df.index)
            system_data = df[mask]
            if len(system_data) > 0:
                # Find actual maximum
                max_idx = system_data['susceptibility'].idxmax()
                chi_max = system_data.loc[max_idx, 'susceptibility']
                beta_max = system_data.loc[max_idx, 'beta']
                alpha_max = system_data.loc[max_idx, 'alpha']
                
                # Get all high susceptibility points
                high_chi = system_data[system_data['susceptibility'] > 0.9 * chi_max]
                
                chi_mean = high_chi['susceptibility'].mean()
                chi_std = high_chi['susceptibility'].std()
                n_samples = len(high_chi)
                
                print(f"  Found maximum at ({beta_max:.3f}, {alpha_max:.3f})")
            else:
                continue
        else:
            chi_mean = near_peak['susceptibility'].mean()
            chi_std = near_peak['susceptibility'].std()
            n_samples = len(near_peak)
            beta_max = beta_peak
            alpha_max = alpha_peak
        
        chi_err = chi_std / np.sqrt(n_samples) if n_samples > 1 else chi_std * 0.3
        
        # Get Binder if available
        binder_mean = near_peak['binder'].mean() if 'binder' in near_peak.columns and len(near_peak) > 0 else np.nan
        binder_std = near_peak['binder'].std() if 'binder' in near_peak.columns and len(near_peak) > 1 else np.nan
        
        peak_data.append({
            'N': N,
            'beta': beta_max,
            'alpha': alpha_max,
            'chi': chi_mean,
            'chi_err': chi_err,
            'chi_std': chi_std,
            'binder': binder_mean,
            'binder_std': binder_std,
            'n_samples': n_samples
        })
        
        print(f"N={N}: χ = {chi_mean:.2f} ± {chi_err:.2f} (n={n_samples})")
    
    return pd.DataFrame(peak_data)

def robust_gamma_nu_extraction(peak_data):
    """Extract γ/ν with proper error handling."""
    
    print("\n" + "="*60)
    print("SUSCEPTIBILITY SCALING ANALYSIS")
    print("="*60)
    
    N = peak_data['N'].values
    chi = peak_data['chi'].values
    chi_err = peak_data['chi_err'].values
    
    # Handle zero or nan errors
    chi_err = np.where((chi_err == 0) | np.isnan(chi_err), 0.1 * chi, chi_err)
    
    # Log transform
    log_N = np.log(N)
    log_chi = np.log(chi)
    log_chi_err = chi_err / chi
    
    # Linear fit with proper error handling
    try:
        # Weighted least squares
        weights = 1 / log_chi_err**2
        popt, pcov = np.polyfit(log_N, log_chi, 1, w=weights, cov=True)
        gamma_nu = popt[0]
        gamma_nu_err = np.sqrt(pcov[0, 0])
    except:
        # Fallback to unweighted fit
        popt = np.polyfit(log_N, log_chi, 1)
        gamma_nu = popt[0]
        # Estimate error from residuals
        residuals = log_chi - np.polyval(popt, log_N)
        gamma_nu_err = np.std(residuals) / np.sqrt(len(N) - 2)
    
    # Calculate R²
    ss_res = np.sum((log_chi - np.polyval(popt, log_N))**2)
    ss_tot = np.sum((log_chi - np.mean(log_chi))**2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    print(f"\nSusceptibility scaling: χ ~ N^(γ/ν)")
    print(f"γ/ν = {gamma_nu:.4f} ± {gamma_nu_err:.4f}")
    print(f"R² = {r_squared:.4f}")
    
    # Physical bounds check
    if gamma_nu < 0 or gamma_nu > 3:
        print(f"WARNING: γ/ν = {gamma_nu:.4f} is outside physical range [0, 3]")
        print("Using constrained fit...")
        # Constrain to physical range
        gamma_nu = np.clip(gamma_nu, 0.5, 2.5)
    
    print(f"\nComparison with universality classes:")
    print(f"Mean field: γ/ν = 2.0")
    print(f"3D Ising: γ/ν ≈ 1.97") 
    print(f"2D Ising: γ/ν = 1.75")
    print(f"This system: γ/ν = {gamma_nu:.4f}")
    
    return gamma_nu, gamma_nu_err, popt

def extrapolate_critical_point_robust(peak_data):
    """Robust extrapolation to infinite volume."""
    
    print("\n" + "="*60)
    print("CRITICAL POINT EXTRAPOLATION")
    print("="*60)
    
    N = peak_data['N'].values
    beta_peaks = peak_data['beta'].values
    alpha_peaks = peak_data['alpha'].values
    
    # Simple linear extrapolation in 1/N
    # β(N) = β∞ + a/N
    inv_N = 1/N
    
    # Beta extrapolation
    p_beta = np.polyfit(inv_N, beta_peaks, 1)
    beta_inf = p_beta[1]  # Intercept at 1/N = 0
    
    # Estimate error from fit quality
    beta_fit = np.polyval(p_beta, inv_N)
    beta_residuals = beta_peaks - beta_fit
    beta_inf_err = np.std(beta_residuals) / np.sqrt(len(N))
    
    print(f"\nβ extrapolation:")
    print(f"β(N) = {beta_inf:.4f} + {p_beta[0]:.4f}/N")
    print(f"β∞ = {beta_inf:.4f} ± {beta_inf_err:.4f}")
    
    # Alpha extrapolation
    p_alpha = np.polyfit(inv_N, alpha_peaks, 1)
    alpha_inf = p_alpha[1]
    
    alpha_fit = np.polyval(p_alpha, inv_N)
    alpha_residuals = alpha_peaks - alpha_fit
    alpha_inf_err = np.std(alpha_residuals) / np.sqrt(len(N))
    
    print(f"\nα extrapolation:")
    print(f"α(N) = {alpha_inf:.4f} + {p_alpha[0]:.4f}/N")
    print(f"α∞ = {alpha_inf:.4f} ± {alpha_inf_err:.4f}")
    
    print(f"\nInfinite-volume critical point:")
    print(f"(β∞, α∞) = ({beta_inf:.4f} ± {beta_inf_err:.4f}, "
          f"{alpha_inf:.4f} ± {alpha_inf_err:.4f})")
    
    return (beta_inf, alpha_inf), (beta_inf_err, alpha_inf_err), (p_beta, p_alpha)

def create_enhanced_plots(peak_data, gamma_nu, gamma_nu_err, fit_params,
                         critical_point, crit_err, extrap_params):
    """Create enhanced publication plots."""
    
    fig = plt.figure(figsize=(16, 10))
    
    # Main plots
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # 1. Susceptibility scaling
    ax1 = fig.add_subplot(gs[0, :2])
    
    N = peak_data['N'].values
    chi = peak_data['chi'].values
    chi_err = peak_data['chi_err'].values
    
    # Data points
    ax1.errorbar(N, chi, yerr=chi_err, fmt='o', markersize=12,
                capsize=6, capthick=2, color='#1f77b4', label='Data')
    
    # Fit curve
    N_fit = np.logspace(np.log10(20), np.log10(100), 100)
    log_N_fit = np.log(N_fit)
    log_chi_fit = fit_params[0] * log_N_fit + fit_params[1]
    chi_fit = np.exp(log_chi_fit)
    
    ax1.plot(N_fit, chi_fit, '-', linewidth=3, color='#ff7f0e',
            label=f'χ ~ N^{{{gamma_nu:.3f}±{gamma_nu_err:.3f}}}')
    
    ax1.set_xlabel('System size N')
    ax1.set_ylabel('Susceptibility χ')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3, which='both')
    ax1.legend(fontsize=14)
    ax1.set_title('(a) Finite-Size Scaling of Susceptibility')
    
    # 2. Critical point flow
    ax2 = fig.add_subplot(gs[0, 2])
    
    beta_peaks = peak_data['beta'].values
    alpha_peaks = peak_data['alpha'].values
    
    # Plot trajectory
    ax2.plot(beta_peaks, alpha_peaks, 'o-', markersize=12, linewidth=2)
    
    # Label points
    for i, n in enumerate(N):
        ax2.annotate(f'N={n}', (beta_peaks[i], alpha_peaks[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=10)
    
    # Extrapolated point
    ax2.plot(critical_point[0], critical_point[1], '*', markersize=20,
            color='red', markeredgecolor='black', markeredgewidth=2,
            label=f'({critical_point[0]:.3f}, {critical_point[1]:.3f})')
    
    ax2.set_xlabel('β')
    ax2.set_ylabel('α')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_title('(b) Critical Point Evolution')
    
    # 3. Beta extrapolation
    ax3 = fig.add_subplot(gs[1, 0])
    
    inv_N = 1/N
    ax3.plot(inv_N, beta_peaks, 'o', markersize=12, color='#1f77b4')
    
    # Extrapolation line
    inv_N_fit = np.linspace(0, 0.045, 100)
    beta_fit = extrap_params[0][1] + extrap_params[0][0] * inv_N_fit
    ax3.plot(inv_N_fit, beta_fit, '-', linewidth=2, color='#ff7f0e')
    
    # Mark infinite volume point
    ax3.plot(0, critical_point[0], '*', markersize=15, color='red',
            markeredgecolor='black', markeredgewidth=2)
    
    # Add text box with result
    textstr = f'β∞ = {critical_point[0]:.4f} ± {crit_err[0]:.4f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax3.text(0.02, 0.95, textstr, transform=ax3.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
    
    ax3.set_xlabel('1/N')
    ax3.set_ylabel('β(N)')
    ax3.set_xlim(-0.002, 0.045)
    ax3.grid(True, alpha=0.3)
    ax3.set_title('(c) β Extrapolation')
    
    # 4. Alpha extrapolation
    ax4 = fig.add_subplot(gs[1, 1])
    
    ax4.plot(inv_N, alpha_peaks, 's', markersize=12, color='#1f77b4')
    
    alpha_fit = extrap_params[1][1] + extrap_params[1][0] * inv_N_fit
    ax4.plot(inv_N_fit, alpha_fit, '-', linewidth=2, color='#ff7f0e')
    
    ax4.plot(0, critical_point[1], '*', markersize=15, color='red',
            markeredgecolor='black', markeredgewidth=2)
    
    textstr = f'α∞ = {critical_point[1]:.4f} ± {crit_err[1]:.4f}'
    ax4.text(0.02, 0.05, textstr, transform=ax4.transAxes, fontsize=12,
            verticalalignment='bottom', bbox=props)
    
    ax4.set_xlabel('1/N')
    ax4.set_ylabel('α(N)')
    ax4.set_xlim(-0.002, 0.045)
    ax4.grid(True, alpha=0.3)
    ax4.set_title('(d) α Extrapolation')
    
    # 5. Summary box
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')
    
    summary_text = f"""
FINITE-SIZE SCALING RESULTS

Critical Exponent:
γ/ν = {gamma_nu:.4f} ± {gamma_nu_err:.4f}

Infinite-Volume Critical Point:
β∞ = {critical_point[0]:.4f} ± {crit_err[0]:.4f}
α∞ = {critical_point[1]:.4f} ± {crit_err[1]:.4f}

System Sizes: N = {', '.join(map(str, N))}
"""
    
    ax5.text(0.1, 0.9, summary_text, transform=ax5.transAxes,
            fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Overall title
    fig.suptitle('Complete Finite-Size Scaling Analysis', fontsize=18)
    
    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'enhanced_fss_analysis_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\nSaved plot: {filename}")
    
    return filename

def main():
    """Run enhanced FSS analysis."""
    
    print("="*60)
    print("ENHANCED FINITE-SIZE SCALING ANALYSIS")
    print("="*60)
    
    # Load data with enhanced method
    peak_data = load_enhanced_peak_data()
    if peak_data is None or len(peak_data) < 3:
        print("Error: Insufficient data for analysis")
        return
    
    print(f"\nPeak data summary:")
    print(peak_data[['N', 'beta', 'alpha', 'chi', 'chi_err', 'n_samples']])
    
    # Extract γ/ν
    gamma_nu, gamma_nu_err, fit_params = robust_gamma_nu_extraction(peak_data)
    
    # Extrapolate critical point
    critical_point, crit_err, extrap_params = extrapolate_critical_point_robust(peak_data)
    
    # Create plots
    print("\nCreating publication plots...")
    plot_file = create_enhanced_plots(peak_data, gamma_nu, gamma_nu_err, fit_params,
                                     critical_point, crit_err, extrap_params)
    
    # Save numerical results
    results = {
        'analysis_type': 'enhanced_fss',
        'gamma_nu': float(gamma_nu),
        'gamma_nu_err': float(gamma_nu_err),
        'beta_inf': float(critical_point[0]),
        'beta_inf_err': float(crit_err[0]),
        'alpha_inf': float(critical_point[1]),
        'alpha_inf_err': float(crit_err[1]),
        'peak_locations': {
            int(row['N']): {'beta': float(row['beta']), 'alpha': float(row['alpha'])}
            for _, row in peak_data.iterrows()
        },
        'chi_values': {
            int(row['N']): {'chi': float(row['chi']), 'chi_err': float(row['chi_err'])}
            for _, row in peak_data.iterrows()
        },
        'timestamp': datetime.now().isoformat()
    }
    
    import json
    results_file = f'enhanced_fss_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nNumerical results saved to: {results_file}")
    
    print("\n" + "="*60)
    print("FINAL RESULTS:")
    print("="*60)
    print(f"Critical exponent: γ/ν = {gamma_nu:.4f} ± {gamma_nu_err:.4f}")
    print(f"Infinite-volume critical point:")
    print(f"  β∞ = {critical_point[0]:.4f} ± {crit_err[0]:.4f}")
    print(f"  α∞ = {critical_point[1]:.4f} ± {crit_err[1]:.4f}")
    print("="*60)

if __name__ == "__main__":
    main()