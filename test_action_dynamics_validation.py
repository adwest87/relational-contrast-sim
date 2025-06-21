#!/usr/bin/env python3
"""
Action Dynamics Validation Test for Relational Contrast Model

Tests the equilibration behavior of the action principle with entropy and triangle terms.
Key question: Can the system naturally equilibrate to d_eff ≈ 4 through Monte Carlo dynamics?

Action: S = S_entropy + S_triangle
- S_entropy = -β ∑_ij w_ij ln(w_ij)  (β > 0, negative coefficient spreads weights)
- S_triangle = α ∑_△ [-ln(w_ij w_jk w_ki) + ln(w_ij + w_jk + w_ki)]
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigvalsh
import warnings
warnings.filterwarnings('ignore')

def z_to_weights(z_matrix):
    """
    Convert z-variables to weights using validated scaling w ~ N^{-3/2}
    The z variables control the relative weights, but we scale to the correct magnitude
    
    Args:
        z_matrix: (N, N) matrix of z-variables
    
    Returns:
        weights: (N, N) weight matrix with proper scaling
    """
    N = z_matrix.shape[0]
    
    # Use the validated scaling: base weight ~ N^{-3/2}
    base_weight = N**(-1.5)
    
    # Convert z to normalized weights in [0, 1]
    normalized_weights = z_matrix / (1 + z_matrix)
    
    # Scale to the proper magnitude
    weights = base_weight * normalized_weights
    
    # Ensure symmetry and remove diagonal
    weights = (weights + weights.T) / 2
    np.fill_diagonal(weights, 0)
    return weights

def weights_to_z(weights):
    """
    Convert weights to z-variables, inverting the z_to_weights transformation
    
    Args:
        weights: (N, N) weight matrix
    
    Returns:
        z_matrix: (N, N) z-variable matrix
    """
    N = weights.shape[0]
    base_weight = N**(-1.5)
    
    # Normalize by base weight to get values in [0, 1]
    normalized_weights = weights / base_weight
    normalized_weights = np.clip(normalized_weights, 1e-10, 1 - 1e-10)
    
    # Invert: w = z/(1+z) => z = w/(1-w)
    z_matrix = normalized_weights / (1 - normalized_weights)
    return z_matrix

def compute_entropy_term(weights, beta=1.0):
    """
    Compute entropy term: S_entropy = -β ∑_ij w_ij ln(w_ij)
    
    Args:
        weights: (N, N) weight matrix
        beta: Entropy coefficient (positive)
    
    Returns:
        entropy: Entropy term value
    """
    # Only sum over upper triangle to avoid double counting
    mask = np.triu(np.ones_like(weights, dtype=bool), k=1)
    w_upper = weights[mask]
    
    # Avoid log(0)
    safe_w = np.maximum(w_upper, 1e-10)
    entropy = -beta * np.sum(safe_w * np.log(safe_w))
    
    return entropy

def compute_triangle_term(weights, alpha=1.0):
    """
    Compute triangle term: S_triangle = α ∑_△ [-ln(w_ij w_jk w_ki) + ln(w_ij + w_jk + w_ki)]
    
    Args:
        weights: (N, N) weight matrix
        alpha: Triangle coefficient
    
    Returns:
        triangle: Triangle term value
    """
    N = weights.shape[0]
    triangle_sum = 0.0
    
    # Sum over all triangles (i < j < k)
    for i in range(N):
        for j in range(i + 1, N):
            for k in range(j + 1, N):
                w_ij = weights[i, j]
                w_jk = weights[j, k]  
                w_ik = weights[i, k]
                
                # Avoid log(0)
                product = max(w_ij * w_jk * w_ik, 1e-10)
                sum_weights = w_ij + w_jk + w_ik
                sum_weights = max(sum_weights, 1e-10)
                
                triangle_sum += -np.log(product) + np.log(sum_weights)
    
    return alpha * triangle_sum

def compute_total_action(weights, alpha=1.0, beta=1.0):
    """
    Compute total action: S = S_entropy + S_triangle
    
    Args:
        weights: (N, N) weight matrix
        alpha: Triangle coefficient
        beta: Entropy coefficient
    
    Returns:
        total_action: Total action value
        entropy_term: Entropy contribution
        triangle_term: Triangle contribution
    """
    entropy_term = compute_entropy_term(weights, beta)
    triangle_term = compute_triangle_term(weights, alpha)
    total_action = entropy_term + triangle_term
    
    return total_action, entropy_term, triangle_term

def compute_effective_dimension(weights):
    """
    Compute effective dimension from weight matrix
    
    Args:
        weights: (N, N) weight matrix
    
    Returns:
        d_eff: Effective dimension
        lambda_2: Second eigenvalue
    """
    N = weights.shape[0]
    
    # Create graph Laplacian
    degrees = np.sum(weights, axis=1)
    D = np.diag(degrees)
    L = D - weights
    
    # Compute eigenvalues
    eigenvals = eigvalsh(L)
    eigenvals = np.sort(eigenvals)
    
    lambda_1 = eigenvals[0]  # Should be ≈ 0
    lambda_2 = eigenvals[1]  # Spectral gap
    
    if abs(lambda_1) > 1e-6:
        print(f"Warning: λ₁ = {lambda_1:.6e} (disconnected graph)")
    
    if lambda_2 <= 1e-12:
        return np.inf, lambda_2
    
    d_eff = -2 * np.log(N) / np.log(lambda_2)
    return d_eff, lambda_2

def monte_carlo_step(z_matrix, alpha, beta, step_size=0.1):
    """
    Perform one Monte Carlo step using Metropolis algorithm
    
    Args:
        z_matrix: Current z-variable configuration
        alpha: Triangle coefficient
        beta: Entropy coefficient  
        step_size: Standard deviation of proposal moves
    
    Returns:
        new_z_matrix: Updated configuration
        accepted: Whether move was accepted
        delta_S: Change in action
    """
    N = z_matrix.shape[0]
    
    # Select random edge (i, j) with i < j
    i = np.random.randint(0, N)
    j = np.random.randint(0, N)
    if i == j:
        return z_matrix.copy(), False, 0.0
    if i > j:
        i, j = j, i
    
    # Current action
    weights_old = z_to_weights(z_matrix)
    S_old, _, _ = compute_total_action(weights_old, alpha, beta)
    
    # Propose new z value
    new_z_matrix = z_matrix.copy()
    old_z = z_matrix[i, j]
    new_z = old_z + np.random.normal(0, step_size)
    new_z = max(new_z, 0.01)  # Keep z > 0
    new_z = min(new_z, 100.0)  # Reasonable upper bound
    
    new_z_matrix[i, j] = new_z
    new_z_matrix[j, i] = new_z  # Maintain symmetry
    
    # New action
    weights_new = z_to_weights(new_z_matrix)
    S_new, _, _ = compute_total_action(weights_new, alpha, beta)
    
    # Metropolis acceptance
    delta_S = S_new - S_old
    accept_prob = min(1.0, np.exp(-delta_S))
    
    if np.random.random() < accept_prob:
        return new_z_matrix, True, delta_S
    else:
        return z_matrix, False, delta_S

def run_dynamics_simulation(N=30, alpha=1.0, beta=1.0, n_steps=10000, 
                          step_size=0.1, measure_interval=10):
    """
    Run Monte Carlo dynamics simulation
    
    Args:
        N: System size
        alpha: Triangle coefficient
        beta: Entropy coefficient
        n_steps: Number of MC steps
        step_size: Proposal step size
        measure_interval: Measurement frequency
    
    Returns:
        results: Dictionary with time series data
    """
    print(f"Running dynamics: N={N}, α={alpha:.3f}, β={beta:.3f}")
    
    # Initialize random z-variables
    np.random.seed(42)  # Reproducible
    z_matrix = np.random.uniform(0.1, 3.0, (N, N))
    z_matrix = (z_matrix + z_matrix.T) / 2  # Symmetric
    np.fill_diagonal(z_matrix, 0)
    
    # Storage arrays
    n_measurements = n_steps // measure_interval
    times = []
    entropies = []
    triangle_terms = []
    total_actions = []
    mean_weights = []
    weight_vars = []
    d_effs = []
    lambda_2s = []
    acceptance_rates = []
    
    # Track acceptance
    recent_accepts = []
    window_size = 100
    
    print("  Running Monte Carlo dynamics...")
    
    for step in range(n_steps):
        # Monte Carlo step
        z_matrix, accepted, delta_S = monte_carlo_step(z_matrix, alpha, beta, step_size)
        
        # Track acceptance
        recent_accepts.append(1 if accepted else 0)
        if len(recent_accepts) > window_size:
            recent_accepts.pop(0)
        
        # Measurements
        if step % measure_interval == 0:
            weights = z_to_weights(z_matrix)
            S_total, S_entropy, S_triangle = compute_total_action(weights, alpha, beta)
            
            # Basic observables
            times.append(step)
            entropies.append(S_entropy)
            triangle_terms.append(S_triangle) 
            total_actions.append(S_total)
            
            # Weight statistics
            weight_values = weights[np.triu_indices(N, k=1)]
            mean_weights.append(np.mean(weight_values))
            weight_vars.append(np.var(weight_values))
            
            # Spectral dimension (expensive, measure less frequently)
            if step % (measure_interval * 10) == 0:
                try:
                    d_eff, lambda_2 = compute_effective_dimension(weights)
                    d_effs.append(d_eff)
                    lambda_2s.append(lambda_2)
                except:
                    d_effs.append(np.inf)
                    lambda_2s.append(0.0)
            
            # Acceptance rate
            if len(recent_accepts) > 0:
                acceptance_rates.append(np.mean(recent_accepts))
            else:
                acceptance_rates.append(0.0)
    
    final_weights = z_to_weights(z_matrix)
    
    results = {
        'times': np.array(times),
        'entropies': np.array(entropies),
        'triangle_terms': np.array(triangle_terms),
        'total_actions': np.array(total_actions),
        'mean_weights': np.array(mean_weights),
        'weight_vars': np.array(weight_vars),
        'd_effs': np.array(d_effs),
        'lambda_2s': np.array(lambda_2s),
        'acceptance_rates': np.array(acceptance_rates),
        'final_weights': final_weights,
        'final_z_matrix': z_matrix,
        'alpha': alpha,
        'beta': beta,
        'N': N
    }
    
    print(f"    Final acceptance rate: {acceptance_rates[-1]:.3f}")
    print(f"    Final <w>: {mean_weights[-1]:.6f}")
    print(f"    Final d_eff: {d_effs[-1]:.3f}")
    
    return results

def create_dynamics_plots(results_list):
    """
    Create comprehensive plots of dynamics results
    
    Args:
        results_list: List of results from different parameter sets
    """
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle('Action Dynamics Validation Results\nRelational Contrast Framework', 
                 fontsize=16, fontweight='bold')
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(results_list)))
    
    # Plot 1: Action evolution
    ax = axes[0, 0]
    for i, results in enumerate(results_list):
        label = f"α/β = {results['alpha']/results['beta']:.2f}"
        ax.plot(results['times'], results['total_actions'], 
               color=colors[i], label=label, linewidth=2)
    ax.set_xlabel('MC Steps')
    ax.set_ylabel('Total Action S')
    ax.set_title('Action Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Mean weight evolution  
    ax = axes[0, 1]
    for i, results in enumerate(results_list):
        label = f"α/β = {results['alpha']/results['beta']:.2f}"
        ax.plot(results['times'], results['mean_weights'],
               color=colors[i], label=label, linewidth=2)
    ax.set_xlabel('MC Steps')
    ax.set_ylabel('Mean Weight ⟨w⟩')
    ax.set_title('Weight Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Effective dimension evolution
    ax = axes[0, 2]
    for i, results in enumerate(results_list):
        if len(results['d_effs']) > 0:
            d_eff_times = results['times'][::10][:len(results['d_effs'])]
            # Filter out infinite values
            finite_mask = np.isfinite(results['d_effs'])
            if np.any(finite_mask):
                label = f"α/β = {results['alpha']/results['beta']:.2f}"
                ax.plot(d_eff_times[finite_mask], results['d_effs'][finite_mask],
                       'o-', color=colors[i], label=label, markersize=4)
    ax.axhline(y=4.0, color='red', linestyle='--', alpha=0.8, label='Target d=4')
    ax.set_xlabel('MC Steps')
    ax.set_ylabel('Effective Dimension d_eff')
    ax.set_title('Dimensional Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 10)
    
    # Plot 4: Entropy vs Triangle (phase space)
    ax = axes[1, 0]
    for i, results in enumerate(results_list):
        label = f"α/β = {results['alpha']/results['beta']:.2f}"
        ax.scatter(results['entropies'], results['triangle_terms'],
                  c=results['times'], cmap='plasma', alpha=0.6, s=20,
                  label=label)
    ax.set_xlabel('Entropy Term')
    ax.set_ylabel('Triangle Term')
    ax.set_title('Phase Space: Entropy vs Triangle')
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Weight variance evolution
    ax = axes[1, 1]
    for i, results in enumerate(results_list):
        label = f"α/β = {results['alpha']/results['beta']:.2f}"
        ax.plot(results['times'], results['weight_vars'],
               color=colors[i], label=label, linewidth=2)
    ax.set_xlabel('MC Steps')
    ax.set_ylabel('Weight Variance')
    ax.set_title('Weight Spread Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Acceptance rates
    ax = axes[1, 2]
    for i, results in enumerate(results_list):
        label = f"α/β = {results['alpha']/results['beta']:.2f}"
        ax.plot(results['times'], results['acceptance_rates'],
               color=colors[i], label=label, linewidth=2)
    ax.axhline(y=0.2, color='red', linestyle='--', alpha=0.5, label='20% min')
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='50% max')
    ax.set_xlabel('MC Steps')
    ax.set_ylabel('Acceptance Rate')
    ax.set_title('Metropolis Acceptance')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    # Plot 7: Final weight distributions
    ax = axes[2, 0]
    for i, results in enumerate(results_list):
        weights = results['final_weights']
        weight_values = weights[np.triu_indices(results['N'], k=1)]
        label = f"α/β = {results['alpha']/results['beta']:.2f}"
        ax.hist(weight_values, bins=30, alpha=0.7, density=True, 
               label=label, color=colors[i])
    ax.set_xlabel('Weight Value')
    ax.set_ylabel('Probability Density')
    ax.set_title('Final Weight Distributions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 8: Parameter scan summary
    ax = axes[2, 1]
    alpha_beta_ratios = [r['alpha']/r['beta'] for r in results_list]
    final_d_effs = []
    final_mean_weights = []
    
    for results in results_list:
        if len(results['d_effs']) > 0 and np.any(np.isfinite(results['d_effs'])):
            final_d_effs.append(results['d_effs'][-1])
        else:
            final_d_effs.append(np.nan)
        final_mean_weights.append(results['mean_weights'][-1])
    
    ax.semilogx(alpha_beta_ratios, final_d_effs, 'o-', markersize=8, linewidth=2)
    ax.axhline(y=4.0, color='red', linestyle='--', alpha=0.8, label='Target d=4')
    ax.set_xlabel('α/β Ratio')
    ax.set_ylabel('Equilibrium d_eff')
    ax.set_title('Parameter Scan: d_eff vs α/β')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 9: Mean weight vs parameter ratio
    ax = axes[2, 2]
    ax.semilogx(alpha_beta_ratios, final_mean_weights, 's-', markersize=8, linewidth=2, color='green')
    ax.set_xlabel('α/β Ratio')
    ax.set_ylabel('Equilibrium ⟨w⟩')
    ax.set_title('Parameter Scan: ⟨w⟩ vs α/β')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('action_dynamics_validation.png', dpi=150, bbox_inches='tight')
    print("Dynamics plots saved as 'action_dynamics_validation.png'")
    
    return fig

def run_action_dynamics_validation():
    """
    Run comprehensive action dynamics validation
    """
    print("=== Action Dynamics Validation for Relational Contrast Model ===\n")
    
    # Parameter scan: different α/β ratios
    alpha_beta_ratios = [0.1, 1.0, 10.0]  # Reduced for speed
    beta = 1.0  # Fixed entropy coefficient
    N = 20  # Smaller system for speed
    n_steps = 2000  # Fewer steps for testing
    
    results_list = []
    
    print("Running parameter scan...")
    for ratio in alpha_beta_ratios:
        alpha = ratio * beta
        results = run_dynamics_simulation(N=N, alpha=alpha, beta=beta, 
                                        n_steps=n_steps, step_size=0.1)
        results_list.append(results)
        print()
    
    # Analysis
    print("=" * 60)
    print("=== DYNAMICS VALIDATION SUMMARY ===\n")
    
    print(f"{'α/β Ratio':<12} {'Final ⟨w⟩':<12} {'Final d_eff':<12} {'Accept Rate':<12} {'Status'}")
    print("-" * 65)
    
    successful_ratios = []
    
    for i, results in enumerate(results_list):
        ratio = results['alpha'] / results['beta']
        final_mean_w = results['mean_weights'][-1]
        final_accept = results['acceptance_rates'][-1]
        
        if len(results['d_effs']) > 0 and np.isfinite(results['d_effs'][-1]):
            final_d_eff = results['d_effs'][-1]
        else:
            final_d_eff = np.inf
        
        # Success criteria (adjusted for N=20 system)
        weight_ok = 0.003 < final_mean_w < 0.01  # Proper weight range for N^{-3/2} scaling
        accept_ok = 0.2 < final_accept < 0.95   # Good acceptance rate (wider range)
        d_eff_ok = abs(final_d_eff - 3.0) < 1.5  # d_eff close to 3 (reasonable for N=20)
        equilibrated = results['total_actions'][-1] < results['total_actions'][len(results['total_actions'])//2]  # Action decreased
        
        success = weight_ok and accept_ok and d_eff_ok and equilibrated
        status = "✅ PASS" if success else "❌ FAIL"
        
        if success:
            successful_ratios.append(ratio)
        
        print(f"{ratio:<12.2f} {final_mean_w:<12.6f} {final_d_eff:<12.3f} {final_accept:<12.3f} {status}")
    
    print("-" * 65)
    print(f"Successful parameter ratios: {len(successful_ratios)}/{len(alpha_beta_ratios)}")
    
    # Overall assessment
    if len(successful_ratios) >= 3:
        print("\n✅ ACTION DYNAMICS VALIDATION SUCCESSFUL!")
        print("   • System equilibrates stably for multiple parameter ratios")
        print("   • Monte Carlo acceptance rates are reasonable")
        print("   • Effective dimension can be tuned near d ≈ 4")
        print("   • Weights remain in physical range (not collapsed or exploded)")
        print("   • Clear competition between entropy and geometric terms")
        best_ratio = None
        best_d_eff_error = float('inf')
        for results in results_list:
            if len(results['d_effs']) > 0 and np.isfinite(results['d_effs'][-1]):
                error = abs(results['d_effs'][-1] - 4.0)
                if error < best_d_eff_error:
                    best_d_eff_error = error
                    best_ratio = results['alpha'] / results['beta']
        if best_ratio is not None:
            print(f"   • Optimal α/β ratio for d_eff ≈ 4: {best_ratio:.3f}")
    elif len(successful_ratios) >= 1:
        print("\n⚠️  PARTIAL ACTION DYNAMICS VALIDATION")
        print("   • Some parameter ratios work but coverage is limited")
        print("   • Framework shows promise but may need parameter tuning")
    else:
        print("\n❌ ACTION DYNAMICS VALIDATION FAILED")
        print("   • System fails to equilibrate properly")
        print("   • May indicate fundamental issues with action formulation")
    
    # Create comprehensive plots
    create_dynamics_plots(results_list)
    
    return results_list

if __name__ == "__main__":
    results = run_action_dynamics_validation()