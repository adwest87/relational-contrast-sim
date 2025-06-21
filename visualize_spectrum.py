#!/usr/bin/env python3
"""
Visualize spectral properties of weighted complete graphs
Demonstrates why uniform complete graphs are poor candidates for emergent spacetime
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
import matplotlib.cm as cm

def complete_graph_laplacian(n, weight_func):
    """Create Laplacian matrix for complete graph with given weight function."""
    L = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i != j:
                w = weight_func(i, j)
                L[i, j] = -w
                L[i, i] += w
    
    return L

def plot_spectrum_comparison():
    """Compare spectra of different weight distributions."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    n = 20
    
    # 1. Uniform weights - degenerate spectrum
    ax = axes[0, 0]
    w_uniform = 0.5
    L = complete_graph_laplacian(n, lambda i, j: w_uniform)
    eigenvalues, _ = eigh(L)
    eigenvalues = np.sort(eigenvalues)
    
    ax.plot(eigenvalues, 'o-', label='Eigenvalues')
    ax.axhline(y=n*w_uniform, color='r', linestyle='--', label=f'Theory: Nw = {n*w_uniform}')
    ax.set_title('Uniform Weights: Degenerate Spectrum')
    ax.set_xlabel('Index')
    ax.set_ylabel('Eigenvalue')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Random weights - lifted degeneracy
    ax = axes[0, 1]
    np.random.seed(42)
    weights = np.random.uniform(0.1, 1.0, (n, n))
    weights = (weights + weights.T) / 2  # Symmetrize
    np.fill_diagonal(weights, 0)
    
    L = complete_graph_laplacian(n, lambda i, j: weights[i, j] if i != j else 0)
    eigenvalues, _ = eigh(L)
    eigenvalues = np.sort(eigenvalues)
    
    ax.plot(eigenvalues, 'o-')
    ax.set_title('Random Weights: Rich Spectrum')
    ax.set_xlabel('Index')
    ax.set_ylabel('Eigenvalue')
    ax.grid(True, alpha=0.3)
    
    # 3. Geometric weights - clustering
    ax = axes[1, 0]
    sigma = 0.5
    
    def geometric_weight(i, j):
        theta_i = 2 * np.pi * i / n
        theta_j = 2 * np.pi * j / n
        d_ij = 2 * np.sin(abs(theta_i - theta_j) / 2)
        return np.exp(-d_ij**2 / sigma**2)
    
    L = complete_graph_laplacian(n, geometric_weight)
    eigenvalues, eigenvectors = eigh(L)
    eigenvalues = np.sort(eigenvalues)
    
    ax.plot(eigenvalues, 'o-')
    ax.set_title(f'Geometric Weights (σ={sigma}): Clustered Spectrum')
    ax.set_xlabel('Index')
    ax.set_ylabel('Eigenvalue')
    ax.grid(True, alpha=0.3)
    
    # 4. Effective dimension vs weight strength
    ax = axes[1, 1]
    n_test = 20
    weights = np.logspace(-3, 0, 50)
    d_effs = []
    
    for w in weights:
        L = complete_graph_laplacian(n_test, lambda i, j: w)
        eigenvalues, _ = eigh(L)
        eigenvalues = np.sort(eigenvalues)
        gap = eigenvalues[1] - eigenvalues[0]
        d_eff = -2 * np.log(n_test) / np.log(gap) if gap > 1e-10 else np.nan
        d_effs.append(d_eff)
    
    ax.semilogx(weights, d_effs, 'b-', linewidth=2)
    ax.axhline(y=4, color='r', linestyle='--', label='d=4 (target)')
    ax.axvline(x=n_test**(-1.5), color='g', linestyle='--', label=f'w=N^(-3/2)={n_test**(-1.5):.3f}')
    ax.set_title('Effective Dimension vs Weight Strength')
    ax.set_xlabel('Uniform Weight w')
    ax.set_ylabel('Effective Dimension')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-10, 10)
    
    plt.tight_layout()
    plt.savefig('spectrum_comparison.png', dpi=150)
    plt.show()

def plot_eigenfunction_structure():
    """Visualize eigenfunctions for different weight patterns."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    n = 30
    
    # Create different weight patterns
    weight_patterns = [
        ('Uniform', lambda i, j: 0.5),
        ('Random', lambda i, j: np.random.uniform(0.1, 1.0)),
        ('Two-cluster', lambda i, j: 0.9 if (i < n//2) == (j < n//2) else 0.1),
        ('Power-law', lambda i, j: 1.0 / (1 + abs(i - j))**1.5),
        ('Geometric', lambda i, j: np.exp(-((i-j)%(n//2))**2 / 25)),
        ('Scale-free hub', lambda i, j: 1.0 if min(i, j) == 0 else 0.1)
    ]
    
    for idx, (name, weight_func) in enumerate(weight_patterns):
        ax = axes[idx // 3, idx % 3]
        
        # Reset random seed for reproducibility
        if 'Random' in name:
            np.random.seed(42)
        
        L = complete_graph_laplacian(n, weight_func)
        eigenvalues, eigenvectors = eigh(L)
        
        # Sort by eigenvalue
        idx_sort = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx_sort]
        eigenvectors = eigenvectors[:, idx_sort]
        
        # Plot weight matrix
        W = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    W[i, j] = weight_func(i, j)
        
        im = ax.imshow(W, cmap='viridis', aspect='auto')
        ax.set_title(f'{name}\nGap={eigenvalues[1]:.3f}')
        ax.set_xlabel('Node j')
        ax.set_ylabel('Node i')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig('weight_patterns.png', dpi=150)
    plt.show()

def analyze_4d_emergence():
    """Detailed analysis of conditions for 4D emergence."""
    plt.figure(figsize=(12, 8))
    
    # Test different graph sizes
    sizes = [10, 20, 50, 100]
    colors = cm.get_cmap('viridis')(np.linspace(0, 1, len(sizes)))
    
    for idx, n in enumerate(sizes):
        weights = np.logspace(-4, 0, 100)
        d_effs = []
        
        for w in weights:
            L = complete_graph_laplacian(n, lambda i, j: w)
            eigenvalues, _ = eigh(L)
            eigenvalues = np.sort(eigenvalues)
            gap = eigenvalues[1] - eigenvalues[0]
            
            if gap > 1e-10:
                d_eff = -2 * np.log(n) / np.log(gap)
            else:
                d_eff = np.nan
            
            d_effs.append(d_eff)
        
        plt.semilogx(weights, d_effs, '-', color=colors[idx], linewidth=2, label=f'N={n}')
        
        # Mark theoretical w for d=4
        w_theory = n**(-1.5)
        plt.axvline(x=w_theory, color=colors[idx], linestyle='--', alpha=0.5)
    
    plt.axhline(y=4, color='red', linestyle='-', linewidth=2, label='d=4 target')
    plt.xlabel('Uniform Weight w', fontsize=12)
    plt.ylabel('Effective Dimension d_eff', fontsize=12)
    plt.title('Effective Dimension vs Weight: Complete Graph', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(-2, 10)
    
    # Add text annotations
    plt.text(0.01, 8, 'Low weights:\nDisconnected\n(d → ∞)', 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.text(0.1, 1, 'High weights:\nSuper-connected\n(d → 0)', 
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('4d_emergence_analysis.png', dpi=150)
    plt.show()

def main():
    """Run all visualizations."""
    print("Generating spectral analysis visualizations...")
    
    plot_spectrum_comparison()
    print("Saved: spectrum_comparison.png")
    
    plot_eigenfunction_structure()
    print("Saved: weight_patterns.png")
    
    analyze_4d_emergence()
    print("Saved: 4d_emergence_analysis.png")
    
    print("\nKey findings visualized:")
    print("1. Uniform complete graphs have trivial (degenerate) spectra")
    print("2. Any weight variation creates rich spectral structure")
    print("3. d_eff = 4 requires precise tuning: w ~ N^(-3/2)")
    print("4. Different weight patterns create different 'geometries'")
    print("5. Complete graphs may be too rigid for realistic spacetime")

if __name__ == "__main__":
    main()