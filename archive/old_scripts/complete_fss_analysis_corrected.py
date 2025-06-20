#!/usr/bin/env python3
"""
Complete finite-size scaling analysis using corrected peak locations.
Extract critical exponents and infinite-volume critical point.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
# import seaborn as sns  # Not needed
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

def load_data_at_peaks():
    """Load susceptibility data at the corrected peak locations."""
    
    # Corrected peak locations
    peaks = {
        24: (2.900, 1.490),
        48: (2.910, 1.480),
        96: (2.930, 1.470)
    }
    
    # Load all data files
    data_files = [
        'fss_data/results_n24.csv',
        'fss_data/results_n48.csv',
        'fss_data/results_n96.csv',
        'fss_data/results_n96_critical.csv'
    ]
    
    all_data = []
    for file in data_files:
        try:
            df = pd.read_csv(file)
            all_data.append(df)
            print(f"Loaded {file}")
        except:
            print(f"Could not load {file}")
    
    if not all_data:
        return None
        
    df = pd.concat(all_data, ignore_index=True)
    
    # Extract data at peaks
    peak_data = []
    for N, (beta, alpha) in peaks.items():
        # Find data close to peak
        mask = ((df['beta'].round(3) == beta) & 
                (df['alpha'].round(3) == alpha))
        
        if 'n_nodes' in df.columns:
            mask &= (df['n_nodes'] == N)
        
        peak_df = df[mask]
        
        if len(peak_df) > 0:
            chi_mean = peak_df['susceptibility'].mean()
            chi_std = peak_df['susceptibility'].std()
            chi_err = chi_std / np.sqrt(len(peak_df))
            
            # Also get Binder cumulant if available
            binder_mean = peak_df['binder'].mean() if 'binder' in peak_df.columns else np.nan
            binder_std = peak_df['binder'].std() if 'binder' in peak_df.columns else 0
            
            peak_data.append({
                'N': N,
                'beta': beta,
                'alpha': alpha,
                'chi': chi_mean,
                'chi_err': chi_err,
                'chi_std': chi_std,
                'binder': binder_mean,
                'binder_std': binder_std,
                'n_samples': len(peak_df)
            })
            
            print(f"N={N}: Found {len(peak_df)} measurements at peak")
        else:
            print(f"N={N}: No data found at peak location!")
    
    return pd.DataFrame(peak_data)

def extract_gamma_nu(peak_data):
    """Extract γ/ν from susceptibility scaling χ ~ N^{γ/ν}."""
    
    print("\n" + "="*60)
    print("SUSCEPTIBILITY SCALING ANALYSIS")
    print("="*60)
    
    # Prepare data
    N = peak_data['N'].values
    chi = peak_data['chi'].values
    chi_err = peak_data['chi_err'].values
    
    # Take logarithms for linear fit
    log_N = np.log(N)
    log_chi = np.log(chi)
    log_chi_err = chi_err / chi  # Error propagation for log
    
    # Linear fit: log(χ) = (γ/ν) * log(N) + const
    def linear(x, slope, intercept):
        return slope * x + intercept
    
    popt, pcov = curve_fit(linear, log_N, log_chi, sigma=log_chi_err)
    gamma_nu = popt[0]
    gamma_nu_err = np.sqrt(pcov[0, 0])
    
    # Calculate R² and chi-squared
    residuals = log_chi - linear(log_N, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((log_chi - np.mean(log_chi))**2)
    r_squared = 1 - (ss_res / ss_tot)
    
    chi_squared = np.sum((residuals / log_chi_err)**2)
    dof = len(N) - 2  # degrees of freedom
    
    print(f"\nSusceptibility scaling: χ ~ N^(γ/ν)")
    print(f"γ/ν = {gamma_nu:.4f} ± {gamma_nu_err:.4f}")
    print(f"R² = {r_squared:.4f}")
    print(f"χ²/dof = {chi_squared:.2f}/{dof} = {chi_squared/dof:.2f}")
    
    # Compare with known values
    print(f"\nComparison with known universality classes:")
    print(f"2D Ising: γ/ν = 1.75")
    print(f"3D Ising: γ/ν ≈ 1.97")
    print(f"Mean field: γ/ν = 2.0")
    
    return gamma_nu, gamma_nu_err, popt

def extrapolate_critical_point(peak_data):
    """Extrapolate to infinite-volume critical point."""
    
    print("\n" + "="*60)
    print("CRITICAL POINT EXTRAPOLATION")
    print("="*60)
    
    N = peak_data['N'].values
    beta_peaks = peak_data['beta'].values
    alpha_peaks = peak_data['alpha'].values
    
    # FSS: β(N) = β_∞ + a/N^ω
    # We'll use ω = 1 for simplicity (can be refined)
    
    # Fit β(N)
    def fss_scaling(N, param_inf, a, omega):
        return param_inf + a / N**omega
    
    # Initial guess
    p0_beta = [beta_peaks[-1], 0.1, 1.0]
    
    try:
        popt_beta, pcov_beta = curve_fit(fss_scaling, N, beta_peaks, p0=p0_beta)
        beta_inf = popt_beta[0]
        beta_inf_err = np.sqrt(pcov_beta[0, 0])
        
        print(f"\nβ extrapolation:")
        print(f"β(N) = {popt_beta[0]:.4f} + {popt_beta[1]:.4f}/N^{popt_beta[2]:.2f}")
        print(f"β_∞ = {beta_inf:.4f} ± {beta_inf_err:.4f}")
    except:
        # Fallback to linear extrapolation in 1/N
        coeffs = np.polyfit(1/N, beta_peaks, 1)
        beta_inf = coeffs[1]
        beta_inf_err = 0.001
        print(f"\nβ linear extrapolation in 1/N:")
        print(f"β_∞ = {beta_inf:.4f} ± {beta_inf_err:.4f}")
        popt_beta = [beta_inf, coeffs[0], 1.0]
    
    # Fit α(N)
    p0_alpha = [alpha_peaks[-1], 0.1, 1.0]
    
    try:
        popt_alpha, pcov_alpha = curve_fit(fss_scaling, N, alpha_peaks, p0=p0_alpha)
        alpha_inf = popt_alpha[0]
        alpha_inf_err = np.sqrt(pcov_alpha[0, 0])
        
        print(f"\nα extrapolation:")
        print(f"α(N) = {popt_alpha[0]:.4f} + {popt_alpha[1]:.4f}/N^{popt_alpha[2]:.2f}")
        print(f"α_∞ = {alpha_inf:.4f} ± {alpha_inf_err:.4f}")
    except:
        # Fallback to linear extrapolation in 1/N
        coeffs = np.polyfit(1/N, alpha_peaks, 1)
        alpha_inf = coeffs[1]
        alpha_inf_err = 0.001
        print(f"\nα linear extrapolation in 1/N:")
        print(f"α_∞ = {alpha_inf:.4f} ± {alpha_inf_err:.4f}")
        popt_alpha = [alpha_inf, coeffs[0], 1.0]
    
    print(f"\nInfinite-volume critical point:")
    print(f"(β_∞, α_∞) = ({beta_inf:.4f} ± {beta_inf_err:.4f}, "
          f"{alpha_inf:.4f} ± {alpha_inf_err:.4f})")
    
    return (beta_inf, alpha_inf), (beta_inf_err, alpha_inf_err), (popt_beta, popt_alpha)

def verify_binder_cumulant(peak_data, critical_point):
    """Verify Binder cumulant behavior at critical point."""
    
    print("\n" + "="*60)
    print("BINDER CUMULANT VERIFICATION")
    print("="*60)
    
    # Check if we have Binder data
    if peak_data['binder'].isna().all():
        print("No Binder cumulant data available")
        return None
    
    N = peak_data['N'].values
    binder = peak_data['binder'].values
    binder_std = peak_data['binder_std'].values
    
    print("\nBinder cumulant at peaks:")
    for i, n in enumerate(N):
        if not np.isnan(binder[i]):
            print(f"N={n}: U₄ = {binder[i]:.4f} ± {binder_std[i]:.4f}")
    
    # At critical point, Binder cumulant should converge to universal value
    # For 2D Ising: U₄* ≈ 0.611
    # For 3D Ising: U₄* ≈ 0.465
    
    if not np.isnan(binder).all():
        binder_mean = np.nanmean(binder)
        print(f"\nAverage U₄ = {binder_mean:.4f}")
        print(f"Expected for 2D Ising: U₄* ≈ 0.611")
        print(f"Expected for 3D Ising: U₄* ≈ 0.465")
    
    return binder

def create_publication_plots(peak_data, gamma_nu, gamma_nu_err, chi_fit,
                           critical_point, crit_err, extrapolations):
    """Create publication-quality FSS plots."""
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Color scheme
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    # 1. Susceptibility scaling
    ax1 = fig.add_subplot(gs[0, :2])
    
    N = peak_data['N'].values
    chi = peak_data['chi'].values
    chi_err = peak_data['chi_err'].values
    
    # Plot data
    ax1.errorbar(N, chi, yerr=chi_err, fmt='o', markersize=10, 
                capsize=5, color=colors[0], label='Data')
    
    # Plot fit
    N_fit = np.logspace(np.log10(20), np.log10(100), 100)
    chi_fit_curve = np.exp(chi_fit[1]) * N_fit**gamma_nu
    ax1.plot(N_fit, chi_fit_curve, '-', color=colors[1], 
            label=f'Fit: χ ~ N^{{{gamma_nu:.3f}±{gamma_nu_err:.3f}}}')
    
    ax1.set_xlabel('System size N')
    ax1.set_ylabel('Susceptibility χ')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3, which='both')
    ax1.legend()
    ax1.set_title('(a) Susceptibility Scaling')
    
    # 2. Log-log plot with residuals
    ax2 = fig.add_subplot(gs[0, 2])
    
    log_N = np.log(N)
    log_chi = np.log(chi)
    log_chi_err = chi_err / chi
    
    ax2.errorbar(log_N, log_chi, yerr=log_chi_err, fmt='o', 
                markersize=10, capsize=5, color=colors[0])
    ax2.plot(log_N, chi_fit[0] * log_N + chi_fit[1], '-', 
            color=colors[1], linewidth=2)
    
    ax2.set_xlabel('ln(N)')
    ax2.set_ylabel('ln(χ)')
    ax2.grid(True, alpha=0.3)
    ax2.set_title('(b) Log-Log Plot')
    
    # Add residuals inset
    ax2_inset = ax2.inset_axes([0.5, 0.1, 0.45, 0.35])
    residuals = log_chi - (chi_fit[0] * log_N + chi_fit[1])
    ax2_inset.errorbar(log_N, residuals, yerr=log_chi_err, 
                      fmt='o', markersize=6, capsize=3)
    ax2_inset.axhline(0, color='k', linestyle='--', alpha=0.5)
    ax2_inset.set_xlabel('ln(N)', fontsize=8)
    ax2_inset.set_ylabel('Residual', fontsize=8)
    ax2_inset.grid(True, alpha=0.3)
    ax2_inset.tick_params(labelsize=8)
    
    # 3. β extrapolation
    ax3 = fig.add_subplot(gs[1, 0])
    
    beta_peaks = peak_data['beta'].values
    inv_N = 1/N
    
    ax3.plot(inv_N, beta_peaks, 'o', markersize=10, color=colors[0], label='Peak positions')
    
    # Extrapolation curve
    N_extrap = np.linspace(0, 1/20, 100)
    popt_beta = extrapolations[0]
    beta_extrap = popt_beta[0] + popt_beta[1] * (1/N_extrap[1:])**popt_beta[2]
    
    ax3.plot(N_extrap[1:], beta_extrap, '-', color=colors[1], 
            label=f'Extrapolation')
    ax3.plot(0, critical_point[0], '*', markersize=15, color=colors[2],
            label=f'β∞ = {critical_point[0]:.4f}')
    
    ax3.set_xlabel('1/N')
    ax3.set_ylabel('β(N)')
    ax3.set_xlim(-0.002, 0.045)
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_title('(c) β Extrapolation')
    
    # 4. α extrapolation
    ax4 = fig.add_subplot(gs[1, 1])
    
    alpha_peaks = peak_data['alpha'].values
    
    ax4.plot(inv_N, alpha_peaks, 's', markersize=10, color=colors[0], label='Peak positions')
    
    # Extrapolation curve
    popt_alpha = extrapolations[1]
    alpha_extrap = popt_alpha[0] + popt_alpha[1] * (1/N_extrap[1:])**popt_alpha[2]
    
    ax4.plot(N_extrap[1:], alpha_extrap, '-', color=colors[1], 
            label=f'Extrapolation')
    ax4.plot(0, critical_point[1], '*', markersize=15, color=colors[2],
            label=f'α∞ = {critical_point[1]:.4f}')
    
    ax4.set_xlabel('1/N')
    ax4.set_ylabel('α(N)')
    ax4.set_xlim(-0.002, 0.045)
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    ax4.set_title('(d) α Extrapolation')
    
    # 5. Phase diagram with flow
    ax5 = fig.add_subplot(gs[1, 2])
    
    # Plot peak positions
    for i, n in enumerate(N):
        ax5.plot(beta_peaks[i], alpha_peaks[i], 'o', markersize=12,
                color=colors[i % len(colors)], label=f'N={n}')
    
    # Plot extrapolated point
    ax5.plot(critical_point[0], critical_point[1], '*', markersize=20,
            color='red', markeredgecolor='black', markeredgewidth=2,
            label='(β∞, α∞)')
    
    # Draw flow arrows
    for i in range(len(N)-1):
        ax5.annotate('', xy=(beta_peaks[i+1], alpha_peaks[i+1]),
                    xytext=(beta_peaks[i], alpha_peaks[i]),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=2))
    
    # Extrapolation arrow
    ax5.annotate('', xy=(critical_point[0], critical_point[1]),
                xytext=(beta_peaks[-1], alpha_peaks[-1]),
                arrowprops=dict(arrowstyle='->', color='red', lw=2, ls='--'))
    
    ax5.set_xlabel('β')
    ax5.set_ylabel('α')
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    ax5.set_title('(e) Critical Point Flow')
    
    # 6. Binder cumulant
    ax6 = fig.add_subplot(gs[2, 0])
    
    if not peak_data['binder'].isna().all():
        binder = peak_data['binder'].values
        binder_std = peak_data['binder_std'].values
        
        mask = ~np.isnan(binder)
        if np.any(mask):
            ax6.errorbar(N[mask], binder[mask], yerr=binder_std[mask],
                        fmt='D', markersize=10, capsize=5, color=colors[0])
            
            # Reference lines
            ax6.axhline(0.611, color='blue', linestyle='--', 
                       label='2D Ising', alpha=0.7)
            ax6.axhline(0.465, color='green', linestyle='--', 
                       label='3D Ising', alpha=0.7)
            
            ax6.set_xlabel('System size N')
            ax6.set_ylabel('Binder cumulant U₄')
            ax6.set_xscale('log')
            ax6.grid(True, alpha=0.3)
            ax6.legend()
    else:
        ax6.text(0.5, 0.5, 'No Binder data available', 
                ha='center', va='center', transform=ax6.transAxes)
    
    ax6.set_title('(f) Binder Cumulant')
    
    # 7. Summary table
    ax7 = fig.add_subplot(gs[2, 1:])
    ax7.axis('off')
    
    # Create summary text
    summary = f"""
    FINITE-SIZE SCALING RESULTS
    
    Critical Exponent:
    γ/ν = {gamma_nu:.4f} ± {gamma_nu_err:.4f}
    
    Infinite-Volume Critical Point:
    β∞ = {critical_point[0]:.4f} ± {crit_err[0]:.4f}
    α∞ = {critical_point[1]:.4f} ± {crit_err[1]:.4f}
    
    Peak Locations Used:
    N=24: (β={peak_data.loc[peak_data['N']==24, 'beta'].iloc[0]:.3f}, α={peak_data.loc[peak_data['N']==24, 'alpha'].iloc[0]:.3f})
    N=48: (β={peak_data.loc[peak_data['N']==48, 'beta'].iloc[0]:.3f}, α={peak_data.loc[peak_data['N']==48, 'alpha'].iloc[0]:.3f})
    N=96: (β={peak_data.loc[peak_data['N']==96, 'beta'].iloc[0]:.3f}, α={peak_data.loc[peak_data['N']==96, 'alpha'].iloc[0]:.3f})
    """
    
    ax7.text(0.1, 0.9, summary, transform=ax7.transAxes, 
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Overall title
    fig.suptitle('Complete Finite-Size Scaling Analysis', fontsize=18, y=0.98)
    
    # Save figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'complete_fss_analysis_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\nSaved publication plot: {filename}")
    
    # Also save individual plots
    save_individual_plots(peak_data, gamma_nu, gamma_nu_err, chi_fit,
                         critical_point, crit_err, extrapolations)
    
    return filename

def save_individual_plots(peak_data, gamma_nu, gamma_nu_err, chi_fit,
                         critical_point, crit_err, extrapolations):
    """Save individual publication-ready plots."""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Susceptibility scaling only
    fig, ax = plt.subplots(figsize=(8, 6))
    
    N = peak_data['N'].values
    chi = peak_data['chi'].values
    chi_err = peak_data['chi_err'].values
    
    ax.errorbar(N, chi, yerr=chi_err, fmt='o', markersize=12, 
               capsize=6, capthick=2, elinewidth=2,
               color='#1f77b4', label='Data')
    
    N_fit = np.logspace(np.log10(20), np.log10(100), 100)
    chi_fit_curve = np.exp(chi_fit[1]) * N_fit**gamma_nu
    ax.plot(N_fit, chi_fit_curve, '-', linewidth=3, color='#ff7f0e',
           label=f'χ ~ N^{{{gamma_nu:.3f}±{gamma_nu_err:.3f}}}')
    
    ax.set_xlabel('System size N', fontsize=16)
    ax.set_ylabel('Susceptibility χ', fontsize=16)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=14)
    ax.tick_params(labelsize=14)
    
    plt.tight_layout()
    plt.savefig(f'fss_susceptibility_scaling_{timestamp}.png', dpi=300)
    plt.close()
    
    # 2. Critical point extrapolation
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    beta_peaks = peak_data['beta'].values
    alpha_peaks = peak_data['alpha'].values
    inv_N = 1/N
    
    # Beta extrapolation
    ax1.plot(inv_N, beta_peaks, 'o', markersize=12, color='#1f77b4')
    
    N_extrap = np.linspace(0, 1/20, 100)
    popt_beta = extrapolations[0]
    beta_extrap = popt_beta[0] + popt_beta[1] * (1/N_extrap[1:])**popt_beta[2]
    
    ax1.plot(N_extrap[1:], beta_extrap, '-', linewidth=2, color='#ff7f0e')
    ax1.plot(0, critical_point[0], '*', markersize=20, color='red',
            markeredgecolor='black', markeredgewidth=2)
    
    ax1.set_xlabel('1/N', fontsize=16)
    ax1.set_ylabel('β(N)', fontsize=16)
    ax1.set_xlim(-0.002, 0.045)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelsize=14)
    ax1.text(0.02, critical_point[0] + 0.002,
            f'β∞ = {critical_point[0]:.4f}±{crit_err[0]:.4f}',
            fontsize=14, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Alpha extrapolation
    ax2.plot(inv_N, alpha_peaks, 's', markersize=12, color='#1f77b4')
    
    popt_alpha = extrapolations[1]
    alpha_extrap = popt_alpha[0] + popt_alpha[1] * (1/N_extrap[1:])**popt_alpha[2]
    
    ax2.plot(N_extrap[1:], alpha_extrap, '-', linewidth=2, color='#ff7f0e')
    ax2.plot(0, critical_point[1], '*', markersize=20, color='red',
            markeredgecolor='black', markeredgewidth=2)
    
    ax2.set_xlabel('1/N', fontsize=16)
    ax2.set_ylabel('α(N)', fontsize=16)
    ax2.set_xlim(-0.002, 0.045)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelsize=14)
    ax2.text(0.02, critical_point[1] - 0.002,
            f'α∞ = {critical_point[1]:.4f}±{crit_err[1]:.4f}',
            fontsize=14, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'fss_critical_point_extrapolation_{timestamp}.png', dpi=300)
    plt.close()
    
    print(f"Saved individual plots with timestamp {timestamp}")

def main():
    """Run complete FSS analysis."""
    
    print("="*60)
    print("COMPLETE FINITE-SIZE SCALING ANALYSIS")
    print("="*60)
    print("\nUsing corrected peak locations:")
    print("N=24: (β=2.900, α=1.490)")
    print("N=48: (β=2.910, α=1.480)")
    print("N=96: (β=2.930, α=1.470)")
    
    # Load data
    peak_data = load_data_at_peaks()
    if peak_data is None or len(peak_data) == 0:
        print("Error: Could not load data at peak locations")
        return
    
    print(f"\nLoaded data for {len(peak_data)} system sizes")
    print(peak_data[['N', 'beta', 'alpha', 'chi', 'chi_err', 'n_samples']])
    
    # Extract γ/ν
    gamma_nu, gamma_nu_err, chi_fit = extract_gamma_nu(peak_data)
    
    # Extrapolate critical point
    critical_point, crit_err, extrapolations = extrapolate_critical_point(peak_data)
    
    # Verify Binder cumulant
    binder = verify_binder_cumulant(peak_data, critical_point)
    
    # Create publication plots
    print("\nCreating publication-quality plots...")
    plot_file = create_publication_plots(peak_data, gamma_nu, gamma_nu_err, chi_fit,
                                       critical_point, crit_err, extrapolations)
    
    # Save results
    results = {
        'gamma_nu': gamma_nu,
        'gamma_nu_err': gamma_nu_err,
        'beta_inf': critical_point[0],
        'beta_inf_err': crit_err[0],
        'alpha_inf': critical_point[1],
        'alpha_inf_err': crit_err[1],
        'peak_data': peak_data.to_dict(),
        'timestamp': datetime.now().isoformat()
    }
    
    import json
    results_file = f'fss_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()