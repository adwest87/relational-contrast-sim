#!/usr/bin/env python3
"""
Scaling Law Validation Test for Relational Contrast Model

Tests the theoretical prediction that effective dimension d_eff = 4 
requires exactly w ~ N^{-3/2} scaling for complete graphs.

Expected: All N values should show d_eff = 4 at α ≈ 1.5
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigvalsh
import warnings
warnings.filterwarnings('ignore')

def create_complete_graph_laplacian(N, w, perturbation_strength=0.01):
    """
    Create weighted complete graph Laplacian with uniform weights + small random perturbation
    
    Args:
        N: Number of nodes
        w: Base weight value
        perturbation_strength: Fraction of w to add as random noise
    
    Returns:
        L: Graph Laplacian matrix
    """
    # Create weight matrix with small random perturbations for numerical stability
    W = np.full((N, N), w)
    
    # Add random perturbations (avoid perfect degeneracy)
    np.random.seed(42)  # Reproducible results
    perturbations = np.random.uniform(-perturbation_strength, perturbation_strength, (N, N))
    perturbations = (perturbations + perturbations.T) / 2  # Make symmetric
    np.fill_diagonal(perturbations, 0)  # No self-loops
    
    W += w * perturbations
    W = np.maximum(W, 1e-10)  # Ensure positive weights
    np.fill_diagonal(W, 0)    # No self-loops
    
    # Create degree matrix
    D = np.diag(np.sum(W, axis=1))
    
    # Graph Laplacian L = D - W
    L = D - W
    
    return L

def compute_effective_dimension(L, N):
    """
    Compute effective dimension d_eff = -2*ln(N)/ln(λ₂)
    
    Args:
        L: Graph Laplacian matrix
        N: Number of nodes
    
    Returns:
        d_eff: Effective dimension
        lambda_2: Second smallest eigenvalue
    """
    # Compute eigenvalues
    eigenvals = eigvalsh(L)
    eigenvals = np.sort(eigenvals)
    
    # Second smallest eigenvalue (λ₁ should be ≈ 0 for connected graphs)
    lambda_1 = eigenvals[0]
    lambda_2 = eigenvals[1]
    
    # Check connectivity (λ₁ should be small)
    if abs(lambda_1) > 1e-8:
        print(f"Warning: λ₁ = {lambda_1:.6e} (graph may be disconnected)")
    
    # Handle edge cases
    if lambda_2 <= 1e-12:
        print(f"Warning: λ₂ = {lambda_2:.6e} too small, setting d_eff = inf")
        return np.inf, lambda_2
    
    # Effective dimension formula
    d_eff = -2 * np.log(N) / np.log(lambda_2)
    
    return d_eff, lambda_2

def run_scaling_validation():
    """
    Run complete scaling law validation test
    """
    print("=== Scaling Law Validation for Relational Contrast Model ===\n")
    
    # Test parameters
    N_values = [20, 30, 40, 50, 60, 80, 100]
    alpha_values = np.linspace(1.0, 2.0, 51)  # α from 1.0 to 2.0
    
    # Storage for results
    results = {}
    closest_alpha_to_4D = {}
    
    print("Testing scaling exponents α from 1.0 to 2.0...")
    print(f"Graph sizes N: {N_values}")
    print(f"Target: d_eff = 4 at α = 1.5\n")
    
    # Run tests for each N
    for N in N_values:
        print(f"Testing N = {N}...")
        
        d_eff_values = []
        lambda_2_values = []
        
        for alpha in alpha_values:
            # Create weights w = N^{-α}
            w = N**(-alpha)
            
            # Create Laplacian
            L = create_complete_graph_laplacian(N, w)
            
            # Compute effective dimension
            d_eff, lambda_2 = compute_effective_dimension(L, N)
            
            d_eff_values.append(d_eff)
            lambda_2_values.append(lambda_2)
        
        # Store results
        results[N] = {
            'alpha': alpha_values,
            'd_eff': np.array(d_eff_values),
            'lambda_2': np.array(lambda_2_values)
        }
        
        # Find α closest to giving d_eff = 4
        finite_mask = np.isfinite(d_eff_values)
        if np.any(finite_mask):
            finite_d_eff = np.array(d_eff_values)[finite_mask]
            finite_alpha = alpha_values[finite_mask]
            
            closest_idx = np.argmin(np.abs(finite_d_eff - 4.0))
            closest_alpha = finite_alpha[closest_idx]
            closest_d_eff = finite_d_eff[closest_idx]
            
            closest_alpha_to_4D[N] = {
                'alpha': closest_alpha,
                'd_eff': closest_d_eff,
                'error': abs(closest_d_eff - 4.0)
            }
            
            print(f"  Closest to d_eff=4: α={closest_alpha:.3f}, d_eff={closest_d_eff:.3f}")
        else:
            print(f"  No finite d_eff values found!")
    
    print("\n" + "="*60)
    
    # Create comprehensive plot
    plt.figure(figsize=(12, 8))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(N_values)))
    
    for i, N in enumerate(N_values):
        data = results[N]
        
        # Plot d_eff vs α
        plt.plot(data['alpha'], data['d_eff'], 
                'o-', color=colors[i], label=f'N={N}', 
                markersize=3, linewidth=1.5)
    
    # Add reference lines
    plt.axhline(y=4.0, color='red', linestyle='--', linewidth=2, 
                label='Target: d_eff = 4', alpha=0.8)
    plt.axvline(x=1.5, color='red', linestyle='--', linewidth=2, 
                label='Theory: α = 1.5', alpha=0.8)
    
    # Formatting
    plt.xlabel('Scaling Exponent α (where w ~ N^{-α})', fontsize=12)
    plt.ylabel('Effective Dimension d_eff', fontsize=12)
    plt.title('Scaling Law Validation: d_eff vs α for Complete Graphs\n' + 
              'Theory predicts: d_eff = 4 at α = 1.5', fontsize=14, pad=20)
    
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Set reasonable y-limits
    plt.ylim(-1, 15)
    plt.xlim(1.0, 2.0)
    
    plt.tight_layout()
    plt.savefig('scaling_law_validation.png', dpi=150, bbox_inches='tight')
    print("Plot saved as 'scaling_law_validation.png'")
    
    # Analysis summary
    print("\n=== ANALYSIS SUMMARY ===")
    print(f"{'N':<6} {'Best α':<8} {'d_eff':<8} {'Error':<8} {'Theory Match'}")
    print("-" * 50)
    
    theory_matches = 0
    alpha_deviations = []
    
    for N in N_values:
        if N in closest_alpha_to_4D:
            data = closest_alpha_to_4D[N]
            alpha_best = data['alpha']
            d_eff_best = data['d_eff']
            error = data['error']
            
            # Check if close to theory (α = 1.5, d_eff = 4)
            alpha_deviation = abs(alpha_best - 1.5)
            is_close = alpha_deviation < 0.1 and error < 0.5
            
            print(f"{N:<6} {alpha_best:<8.3f} {d_eff_best:<8.3f} {error:<8.3f} {'✓' if is_close else '✗'}")
            
            if is_close:
                theory_matches += 1
            alpha_deviations.append(alpha_deviation)
    
    print("-" * 50)
    
    # Overall assessment
    if len(alpha_deviations) > 0:
        mean_alpha_dev = np.mean(alpha_deviations)
        std_alpha_dev = np.std(alpha_deviations)
        
        print(f"\nTheory validation:")
        print(f"  Graphs matching theory: {theory_matches}/{len(N_values)}")
        print(f"  Mean α deviation from 1.5: {mean_alpha_dev:.3f} ± {std_alpha_dev:.3f}")
        
        if theory_matches >= len(N_values) * 0.8:  # 80% threshold
            print(f"  ✅ STRONG EVIDENCE FOR THEORY: α ≈ 1.5 consistently gives d_eff ≈ 4")
        elif theory_matches >= len(N_values) * 0.5:  # 50% threshold
            print(f"  ⚠️  PARTIAL EVIDENCE: Some graphs support theory")
        else:
            print(f"  ❌ WEAK EVIDENCE: Theory not well supported")
            
        if mean_alpha_dev < 0.1:
            print(f"  ✅ EXCELLENT α CONSISTENCY: Crossing point very stable with N")
        elif mean_alpha_dev < 0.2:
            print(f"  ✅ GOOD α CONSISTENCY: Crossing point reasonably stable")
        else:
            print(f"  ⚠️  VARIABLE α: Crossing point varies significantly with N")
    
    print(f"\nConclusion:")
    if theory_matches >= len(N_values) * 0.8 and np.mean(alpha_deviations) < 0.15:
        print("✅ VALIDATES RELATIONAL CONTRAST MODEL:")
        print("   The scaling law w ~ N^{-3/2} robustly produces d_eff ≈ 4")
        print("   This supports the theory that 4D emergence is natural, not fine-tuned")
    else:
        print("⚠️  THEORY NEEDS REFINEMENT:")
        print("   The scaling law prediction shows significant deviations")
        print("   Either the model or the effective dimension formula may need adjustment")

if __name__ == "__main__":
    run_scaling_validation()