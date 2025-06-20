#!/usr/bin/env python3
"""
Advanced critical ridge finder for Relational Contrast phase space.
Uses multiple methods to identify phase transitions in (β, α) space.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate, signal, optimize
from scipy.ndimage import gaussian_filter
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import seaborn as sns

class CriticalRidgeFinder:
    def __init__(self, data_file="scan_results.csv"):
        """Initialize with scan data."""
        self.df = pd.read_csv(data_file)
        self.setup_grids()
        
    def setup_grids(self):
        """Create 2D grids from scan data."""
        # Pivot data into 2D arrays
        self.chi_grid = self.df.pivot(index="beta", columns="alpha", values="susceptibility")
        self.c_grid = self.df.pivot(index="beta", columns="alpha", values="C")
        self.s_bar_grid = self.df.pivot(index="beta", columns="alpha", values="S_bar")
        self.delta_bar_grid = self.df.pivot(index="beta", columns="alpha", values="Delta_bar")
        
        self.alphas = self.chi_grid.columns.values
        self.betas = self.chi_grid.index.values
        
    def find_ridge_gradient(self, observable="C", smooth_sigma=1.0):
        """Find ridge using gradient magnitude."""
        if observable == "C":
            data = self.c_grid.values
        elif observable == "chi":
            data = self.chi_grid.values
        else:
            raise ValueError(f"Unknown observable: {observable}")
            
        # Smooth data
        smoothed = gaussian_filter(data, sigma=smooth_sigma)
        
        # Compute gradient
        gy, gx = np.gradient(smoothed)
        grad_mag = np.sqrt(gx**2 + gy**2)
        
        # Find ridge as maximum gradient for each β
        ridge_alphas = []
        ridge_betas = []
        
        for i, beta in enumerate(self.betas):
            if i < len(grad_mag):
                j_max = np.argmax(grad_mag[i, :])
                ridge_betas.append(beta)
                ridge_alphas.append(self.alphas[j_max])
                
        return np.array(ridge_betas), np.array(ridge_alphas)
    
    def find_ridge_peak(self, observable="C"):
        """Find ridge as peak locations."""
        if observable == "C":
            data = self.c_grid.values
        elif observable == "chi":
            data = self.chi_grid.values
            
        ridge_alphas = []
        ridge_betas = []
        
        for i, beta in enumerate(self.betas):
            # Find peaks in this β slice
            peaks, properties = signal.find_peaks(data[i, :], 
                                                prominence=0.1*np.max(data[i, :]))
            if len(peaks) > 0:
                # Take the most prominent peak
                prominences = properties["prominences"]
                best_peak = peaks[np.argmax(prominences)]
                ridge_betas.append(beta)
                ridge_alphas.append(self.alphas[best_peak])
                
        return np.array(ridge_betas), np.array(ridge_alphas)
    
    def find_ridge_gp(self, observable="C", length_scale=0.1):
        """Use Gaussian Process to find smooth ridge."""
        if observable == "C":
            data = self.c_grid.values
        elif observable == "chi":
            data = self.chi_grid.values
            
        # Flatten data for GP
        X = []
        y = []
        for i, beta in enumerate(self.betas):
            for j, alpha in enumerate(self.alphas):
                X.append([beta, alpha])
                y.append(data[i, j])
                
        X = np.array(X)
        y = np.array(y)
        
        # Fit GP
        kernel = RBF(length_scale=length_scale) + WhiteKernel(noise_level=0.01)
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
        gp.fit(X, y)
        
        # Find ridge by optimizing GP prediction
        ridge_alphas = []
        ridge_betas = []
        
        for beta in self.betas:
            def neg_gp(alpha):
                return -gp.predict([[beta, alpha[0]]])[0]
            
            result = optimize.minimize_scalar(neg_gp, 
                                            bounds=(self.alphas[0], self.alphas[-1]),
                                            method='bounded')
            ridge_betas.append(beta)
            ridge_alphas.append(result.x)
            
        return np.array(ridge_betas), np.array(ridge_alphas)
    
    def find_ridge_balance(self):
        """Find ridge where entropy and gauge terms balance."""
        ridge_alphas = []
        ridge_betas = []
        
        for i, beta in enumerate(self.betas):
            # Look for where |∂S_bar/∂α| ≈ |∂Delta_bar/∂α|
            s_bar = self.s_bar_grid.iloc[i].values
            delta_bar = self.delta_bar_grid.iloc[i].values
            
            # Compute derivatives
            ds_da = np.gradient(s_bar)
            dd_da = np.gradient(delta_bar)
            
            # Find crossing point
            diff = np.abs(ds_da) - np.abs(dd_da)
            sign_changes = np.where(np.diff(np.sign(diff)))[0]
            
            if len(sign_changes) > 0:
                # Interpolate to find exact crossing
                j = sign_changes[0]
                alpha_cross = self.alphas[j] + (self.alphas[j+1] - self.alphas[j]) * \
                              (-diff[j] / (diff[j+1] - diff[j]))
                ridge_betas.append(beta)
                ridge_alphas.append(alpha_cross)
                
        return np.array(ridge_betas), np.array(ridge_alphas)
    
    def fit_critical_scaling(self, beta_ridge, alpha_ridge):
        """Fit power law α ~ β^ν to ridge data."""
        # Use log-log fit
        log_beta = np.log(beta_ridge)
        log_alpha = np.log(alpha_ridge)
        
        # Linear fit in log space
        coeffs = np.polyfit(log_beta, log_alpha, 1)
        nu = coeffs[0]
        A = np.exp(coeffs[1])
        
        return A, nu
    
    def plot_ridge_comparison(self):
        """Compare different ridge-finding methods."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Method 1: Gradient
        beta1, alpha1 = self.find_ridge_gradient("C")
        axes[0,0].imshow(self.c_grid, origin='lower', aspect='auto',
                         extent=[self.alphas[0], self.alphas[-1], 
                                self.betas[0], self.betas[-1]])
        axes[0,0].plot(alpha1, beta1, 'r-', linewidth=2, label='Gradient method')
        axes[0,0].set_title('Specific Heat C - Gradient Ridge')
        axes[0,0].set_xlabel('α')
        axes[0,0].set_ylabel('β')
        axes[0,0].legend()
        
        # Method 2: Peak
        beta2, alpha2 = self.find_ridge_peak("C")
        axes[0,1].imshow(self.c_grid, origin='lower', aspect='auto',
                         extent=[self.alphas[0], self.alphas[-1], 
                                self.betas[0], self.betas[-1]])
        axes[0,1].plot(alpha2, beta2, 'g-', linewidth=2, label='Peak method')
        axes[0,1].set_title('Specific Heat C - Peak Ridge')
        axes[0,1].set_xlabel('α')
        axes[0,1].set_ylabel('β')
        axes[0,1].legend()
        
        # Method 3: Balance
        beta3, alpha3 = self.find_ridge_balance()
        axes[1,0].imshow(self.c_grid, origin='lower', aspect='auto',
                         extent=[self.alphas[0], self.alphas[-1], 
                                self.betas[0], self.betas[-1]])
        axes[1,0].plot(alpha3, beta3, 'b-', linewidth=2, label='Balance method')
        axes[1,0].set_title('Specific Heat C - Balance Ridge')
        axes[1,0].set_xlabel('α')
        axes[1,0].set_ylabel('β')
        axes[1,0].legend()
        
        # All methods together
        axes[1,1].plot(alpha1, beta1, 'r-', linewidth=2, label='Gradient')
        axes[1,1].plot(alpha2, beta2, 'g--', linewidth=2, label='Peak')
        axes[1,1].plot(alpha3, beta3, 'b:', linewidth=2, label='Balance')
        
        # Fit and plot scaling
        if len(beta1) > 3:
            A, nu = self.fit_critical_scaling(beta1, alpha1)
            beta_fit = np.linspace(beta1.min(), beta1.max(), 100)
            alpha_fit = A * beta_fit**nu
            axes[1,1].plot(alpha_fit, beta_fit, 'k--', alpha=0.5, 
                          label=f'α ~ β^{nu:.2f}')
        
        axes[1,1].set_xlabel('α')
        axes[1,1].set_ylabel('β')
        axes[1,1].set_title('Ridge Comparison')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def export_ridge_points(self, method="gradient"):
        """Export ridge points for narrow scan."""
        if method == "gradient":
            betas, alphas = self.find_ridge_gradient("C")
        elif method == "peak":
            betas, alphas = self.find_ridge_peak("C")
        elif method == "balance":
            betas, alphas = self.find_ridge_balance()
        else:
            raise ValueError(f"Unknown method: {method}")
            
        # Create dataframe
        ridge_df = pd.DataFrame({
            'beta': betas,
            'alpha': alphas
        })
        
        # Add nearby points for better sampling
        expanded_points = []
        for _, row in ridge_df.iterrows():
            beta, alpha = row['beta'], row['alpha']
            # Add points in a cross pattern around ridge
            for db in [-0.05, 0, 0.05]:
                for da in [-0.1, 0, 0.1]:
                    if db != 0 or da != 0:
                        expanded_points.append({
                            'beta': beta + db,
                            'alpha': alpha + da
                        })
        
        expanded_df = pd.DataFrame(expanded_points)
        full_df = pd.concat([ridge_df, expanded_df], ignore_index=True)
        
        # Remove duplicates and out-of-bounds points
        full_df = full_df.drop_duplicates()
        full_df = full_df[(full_df['beta'] >= self.betas.min()) & 
                          (full_df['beta'] <= self.betas.max()) &
                          (full_df['alpha'] >= self.alphas.min()) & 
                          (full_df['alpha'] <= self.alphas.max())]
        
        # Save to CSV
        full_df.to_csv('ridge_points.csv', index=False, header=False)
        print(f"Exported {len(full_df)} points to ridge_points.csv")
        
        return full_df
    
    def analyze_scaling_collapse(self):
        """Check for data collapse near critical ridge."""
        # Get ridge
        beta_ridge, alpha_ridge = self.find_ridge_gradient("C")
        
        # Interpolate to get α(β) function
        alpha_func = interpolate.interp1d(beta_ridge, alpha_ridge, 
                                        kind='cubic', fill_value='extrapolate')
        
        # For each point, compute distance from ridge
        distances = []
        c_values = []
        
        for i, beta in enumerate(self.betas):
            alpha_critical = alpha_func(beta)
            for j, alpha in enumerate(self.alphas):
                dist = alpha - alpha_critical
                distances.append(dist)
                c_values.append(self.c_grid.iloc[i, j])
                
        # Try scaling collapse: C * β^a vs (α - α_c) * β^b
        # This is a simplified finite-size scaling analysis
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Group by β and plot
        for beta in self.betas[::2]:  # Every other β for clarity
            mask = np.abs(self.df['beta'] - beta) < 0.01
            subset = self.df[mask]
            
            alpha_c = alpha_func(beta)
            x = (subset['alpha'] - alpha_c) * beta**0.5  # Trial scaling
            y = subset['C'] * beta**0.1  # Trial scaling
            
            ax.plot(x, y, 'o-', alpha=0.6, label=f'β={beta:.2f}')
            
        ax.set_xlabel('(α - α_c) * β^0.5')
        ax.set_ylabel('C * β^0.1')
        ax.set_title('Scaling Collapse Analysis')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        return fig


# Example usage
if __name__ == "__main__":
    # Initialize finder
    finder = CriticalRidgeFinder("scan_results.csv")
    
    # Plot comparison of methods
    fig1 = finder.plot_ridge_comparison()
    plt.savefig("ridge_comparison.png", dpi=300, bbox_inches='tight')
    
    # Export points for narrow scan
    ridge_points = finder.export_ridge_points(method="gradient")
    
    # Analyze scaling
    fig2 = finder.analyze_scaling_collapse()
    plt.savefig("scaling_collapse.png", dpi=300, bbox_inches='tight')
    
    # Print scaling exponent
    beta_ridge, alpha_ridge = finder.find_ridge_gradient("C")
    A, nu = finder.fit_critical_scaling(beta_ridge, alpha_ridge)
    print(f"\nCritical scaling: α ~ {A:.3f} * β^{nu:.3f}")
    
    plt.show()