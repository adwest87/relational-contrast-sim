#!/usr/bin/env python3
"""
Run importance-sampled scan of the critical region.
Uses ridge-biased sampling to efficiently explore near the phase transition.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import subprocess
import os

class ImportanceSampledScan:
    def __init__(self, n_nodes=48):
        self.n_nodes = n_nodes
        self.ridge_slope = 0.06
        self.ridge_intercept = 1.31
        self.ridge_width = 0.02  # Adaptive based on system size
        
        # Adjust width based on finite-size effects
        self.ridge_width *= np.sqrt(24.0 / n_nodes)
        
    def generate_importance_points(self, n_points=100, adaptive=True):
        """Generate points using importance sampling biased toward the ridge."""
        
        points = []
        weights = []
        
        # Beta range based on system size
        beta_min = 2.85 - 0.05 / np.sqrt(self.n_nodes / 24.0)
        beta_max = 2.95 + 0.05 / np.sqrt(self.n_nodes / 24.0)
        
        for i in range(n_points):
            # Sample beta uniformly
            beta = np.random.uniform(beta_min, beta_max)
            
            # Sample alpha from Gaussian around ridge
            alpha_ridge = self.ridge_slope * beta + self.ridge_intercept
            alpha = np.random.normal(alpha_ridge, self.ridge_width)
            
            # Ensure alpha is in reasonable range
            alpha = np.clip(alpha, 1.45, 1.55)
            
            # Calculate importance weight
            # Proposal: Gaussian around ridge
            dist = abs(alpha - alpha_ridge)
            proposal_prob = np.exp(-0.5 * (dist / self.ridge_width)**2) / (self.ridge_width * np.sqrt(2 * np.pi))
            
            # Target: uniform over region
            target_prob = 1.0 / ((beta_max - beta_min) * (1.55 - 1.45))
            
            weight = target_prob / proposal_prob
            
            points.append((beta, alpha))
            weights.append(weight)
            
        return np.array(points), np.array(weights)
    
    def adapt_ridge_parameters(self, results_df):
        """Adapt ridge parameters based on observed susceptibility peaks."""
        
        if len(results_df) < 20:
            return
        
        # Weight by susceptibility
        chi = results_df['susceptibility'].values
        weights = chi / chi.sum()
        
        # Weighted linear regression
        beta = results_df['beta'].values
        alpha = results_df['alpha'].values
        
        # Fit weighted line
        A = np.vstack([beta, np.ones(len(beta))]).T
        slope, intercept = np.linalg.lstsq(A, alpha, rcond=None)[0]
        
        # Smooth adaptation
        self.ridge_slope = 0.8 * self.ridge_slope + 0.2 * slope
        self.ridge_intercept = 0.8 * self.ridge_intercept + 0.2 * intercept
        
        # Adapt width based on scatter
        alpha_pred = self.ridge_slope * beta + self.ridge_intercept
        residuals = alpha - alpha_pred
        self.ridge_width = 0.8 * self.ridge_width + 0.2 * np.std(residuals)
        
        print(f"Adapted ridge: α = {self.ridge_slope:.4f}β + {self.ridge_intercept:.4f}, width = {self.ridge_width:.4f}")
    
    def create_scan_file(self, points, weights, filename='importance_scan_points.csv'):
        """Create CSV file with importance-sampled points and weights."""
        
        with open(filename, 'w') as f:
            f.write("beta,alpha,weight\n")
            for (beta, alpha), weight in zip(points, weights):
                f.write(f"{beta:.4f},{alpha:.4f},{weight:.6f}\n")
        
        return filename
    
    def run_scan(self, n_points=100, mc_steps=100000, n_replicas=10):
        """Run importance-sampled scan with adaptive ridge following."""
        
        print(f"=== Importance-Sampled Critical Region Scan ===")
        print(f"System size: N = {self.n_nodes}")
        print(f"Initial ridge: α = {self.ridge_slope:.3f}β + {self.ridge_intercept:.3f}")
        print(f"Ridge width: {self.ridge_width:.3f}")
        print(f"Points: {n_points}, MC steps: {mc_steps}, Replicas: {n_replicas}")
        
        all_results = []
        
        # Run in batches with adaptation
        batch_size = 20
        n_batches = n_points // batch_size
        
        for batch in range(n_batches):
            print(f"\nBatch {batch + 1}/{n_batches}")
            
            # Generate importance-sampled points
            points, weights = self.generate_importance_points(batch_size)
            
            # Create scan file
            scan_file = f'importance_batch_{batch}.csv'
            self.create_scan_file(points, weights, scan_file)
            
            # Run simulation
            output_file = f'importance_results_N{self.n_nodes}_batch{batch}_{datetime.now():%Y%m%d_%H%M%S}.csv'
            
            cmd = [
                'cargo', 'run', '--release', '--bin', 'fss_narrow_scan', '--',
                '--pairs', scan_file,
                '--output', output_file,
                '--nodes', str(self.n_nodes),
                '--steps', str(mc_steps),
                '--replicas', str(n_replicas),
                '--equilibration', str(mc_steps // 5)
            ]
            
            print(f"Running: {' '.join(cmd)}")
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                print("Simulation completed successfully")
                
                # Load results
                if os.path.exists(output_file):
                    batch_df = pd.read_csv(output_file)
                    
                    # Add importance weights
                    batch_df['importance_weight'] = np.repeat(weights, n_replicas)
                    
                    all_results.append(batch_df)
                    
                    # Adapt ridge parameters
                    combined_df = pd.concat(all_results, ignore_index=True)
                    self.adapt_ridge_parameters(combined_df)
                    
            except subprocess.CalledProcessError as e:
                print(f"Error running simulation: {e}")
                print(f"Output: {e.output}")
        
        # Combine all results
        if all_results:
            final_df = pd.concat(all_results, ignore_index=True)
            final_df.to_csv(f'importance_scan_complete_N{self.n_nodes}_{datetime.now():%Y%m%d_%H%M%S}.csv', index=False)
            
            return final_df
        else:
            return None
    
    def analyze_results(self, df):
        """Analyze importance-sampled results."""
        
        print("\n=== Importance Sampling Analysis ===")
        
        # Group by (beta, alpha) and compute weighted averages
        grouped = df.groupby(['beta', 'alpha']).agg({
            'susceptibility': ['mean', 'std', 'count'],
            'binder': 'mean',
            'mean_action': 'mean',
            'importance_weight': 'first'
        }).reset_index()
        
        # Flatten column names
        grouped.columns = ['beta', 'alpha', 'chi_mean', 'chi_std', 'count', 
                          'binder', 'action', 'weight']
        
        # Normalize weights
        grouped['norm_weight'] = grouped['weight'] / grouped['weight'].sum()
        
        # Effective sample size
        n_eff = 1.0 / (grouped['norm_weight'] ** 2).sum()
        efficiency = n_eff / len(grouped)
        
        print(f"Total points sampled: {len(grouped)}")
        print(f"Effective sample size: {n_eff:.1f}")
        print(f"Sampling efficiency: {efficiency:.1%}")
        
        # Find weighted peak
        weighted_chi = (grouped['chi_mean'] * grouped['norm_weight']).sum()
        peak_idx = grouped['chi_mean'].idxmax()
        peak_beta = grouped.loc[peak_idx, 'beta']
        peak_alpha = grouped.loc[peak_idx, 'alpha']
        peak_chi = grouped.loc[peak_idx, 'chi_mean']
        
        print(f"\nPeak location:")
        print(f"  β = {peak_beta:.4f}")
        print(f"  α = {peak_alpha:.4f}")
        print(f"  χ = {peak_chi:.2f}")
        
        # Ridge fit
        beta_vals = grouped['beta'].values
        alpha_vals = grouped['alpha'].values
        chi_vals = grouped['chi_mean'].values
        
        # Weighted regression
        weights = chi_vals / chi_vals.sum()
        coeffs = np.polyfit(beta_vals, alpha_vals, 1, w=weights)
        
        print(f"\nFitted ridge: α = {coeffs[0]:.4f}β + {coeffs[1]:.4f}")
        
        return grouped
    
    def plot_results(self, df):
        """Create visualization of importance-sampled results."""
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Scatter plot with importance weights
        ax1 = axes[0, 0]
        grouped = df.groupby(['beta', 'alpha']).agg({
            'susceptibility': 'mean',
            'importance_weight': 'first'
        }).reset_index()
        
        scatter = ax1.scatter(grouped['beta'], grouped['alpha'], 
                            c=grouped['susceptibility'], 
                            s=50 / grouped['importance_weight'],  # Size inversely proportional to weight
                            cmap='hot', alpha=0.6)
        
        # Plot ridge
        beta_range = np.linspace(grouped['beta'].min(), grouped['beta'].max(), 100)
        alpha_ridge = self.ridge_slope * beta_range + self.ridge_intercept
        ax1.plot(beta_range, alpha_ridge, 'b--', label='Sampling ridge')
        
        ax1.set_xlabel('β')
        ax1.set_ylabel('α')
        ax1.set_title('Importance-Sampled Points\n(size ∝ 1/weight)')
        plt.colorbar(scatter, ax=ax1, label='χ')
        ax1.legend()
        
        # 2. Weight distribution
        ax2 = axes[0, 1]
        weights = grouped['importance_weight'].values
        ax2.hist(weights, bins=30, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Importance Weight')
        ax2.set_ylabel('Count')
        ax2.set_title('Weight Distribution')
        ax2.axvline(1.0, color='red', linestyle='--', label='Uniform')
        ax2.legend()
        
        # 3. Susceptibility along ridge
        ax3 = axes[1, 0]
        
        # Project points onto ridge
        beta_vals = grouped['beta'].values
        alpha_vals = grouped['alpha'].values
        chi_vals = grouped['susceptibility'].values
        
        # Distance along ridge
        ridge_dist = beta_vals - beta_vals.min()
        
        ax3.scatter(ridge_dist, chi_vals, alpha=0.6)
        ax3.set_xlabel('Distance along ridge')
        ax3.set_ylabel('Susceptibility χ')
        ax3.set_title('χ Profile Along Ridge')
        
        # 4. Efficiency metrics
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Calculate metrics
        n_total = len(grouped)
        n_eff = 1.0 / ((weights / weights.sum()) ** 2).sum()
        efficiency = n_eff / n_total
        
        metrics_text = f"""Importance Sampling Metrics:
        
Total samples: {n_total}
Effective samples: {n_eff:.1f}
Efficiency: {efficiency:.1%}

Weight statistics:
  Min: {weights.min():.3f}
  Mean: {weights.mean():.3f}
  Max: {weights.max():.3f}
  
Ridge parameters:
  Slope: {self.ridge_slope:.4f}
  Intercept: {self.ridge_intercept:.4f}
  Width: {self.ridge_width:.4f}"""
        
        ax4.text(0.1, 0.9, metrics_text, transform=ax4.transAxes,
                verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        # Save figure
        filename = f'importance_sampling_analysis_N{self.n_nodes}_{datetime.now():%Y%m%d_%H%M%S}.png'
        plt.savefig(filename, dpi=300)
        print(f"\nFigure saved: {filename}")
        
        plt.show()

def main():
    """Run importance-sampled critical region scan."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Run importance-sampled critical region scan')
    parser.add_argument('-N', '--nodes', type=int, default=48, help='Number of nodes')
    parser.add_argument('-n', '--npoints', type=int, default=100, help='Number of points to sample')
    parser.add_argument('-s', '--steps', type=int, default=100000, help='MC steps per point')
    parser.add_argument('-r', '--replicas', type=int, default=10, help='Number of replicas')
    
    args = parser.parse_args()
    
    # Create scanner
    scanner = ImportanceSampledScan(args.nodes)
    
    # Run scan
    results = scanner.run_scan(args.npoints, args.steps, args.replicas)
    
    if results is not None:
        # Analyze results
        grouped = scanner.analyze_results(results)
        
        # Plot results
        scanner.plot_results(results)

if __name__ == "__main__":
    main()