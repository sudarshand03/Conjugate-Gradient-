import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm, eigh
from models.conjugate_gradient import conjugate_gradient

def create_clustered_matrix(n=10, perturb=True):
    """Create matrix with 5 clusters of eigenvalues."""
    # Base eigenvalues for 5 clusters
    cluster_centers = [1, 10, 100, 1000, 10000]
    eigenvalues = []
    
    # Create clusters with/without perturbation
    for center in cluster_centers:
        if perturb:
            # Add perturbation within 10% of cluster center
            cluster = [center * (1 + 0.1 * np.random.randn()) for _ in range(2)]
        else:
            # Exact duplicates
            cluster = [center, center]
        eigenvalues.extend(cluster)
    
    # Create orthogonal matrix
    Q, _ = np.linalg.qr(np.random.randn(n, n))
    
    # Create matrix A = QΛQ^T
    A = Q @ np.diag(eigenvalues) @ Q.T
    
    return A, eigenvalues

def create_distinct_matrix(n=10):
    """Create matrix with exactly 5 distinct eigenvalues."""
    # Exactly 5 distinct values
    eigenvalues = [1, 10, 100, 1000, 10000] * 2  # Repeat each twice
    
    # Create orthogonal matrix
    Q, _ = np.linalg.qr(np.random.randn(n, n))
    
    # Create matrix A = QΛQ^T
    A = Q @ np.diag(eigenvalues) @ Q.T
    
    return A, eigenvalues

def run_eigenvalue_experiments():
    """Run experiments with different eigenvalue distributions."""
    n = 10  # Fixed size for all experiments
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Test 1: Clustered eigenvalues (with and without perturbation)
    for perturb in [True, False]:
        A, eigs = create_clustered_matrix(n, perturb)
        b = np.random.randn(n)
        b /= norm(b)
        x0 = np.zeros(n)
        
        # Run CG
        x_final, history, iters, _ = conjugate_gradient(
            A, b, x0, tolerance=1e-10, max_iterations=1000)
        
        # Compute residuals
        residuals = [norm(b - A @ x) for x in history]
        
        # Plot
        label = 'Perturbed Clusters' if perturb else 'Exact Clusters'
        ax1.semilogy(range(len(residuals)), residuals,
                    label=label, marker='.', markevery=5)
    
    ax1.grid(True)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Residual Norm')
    ax1.set_title('CG Convergence: Clustered Eigenvalues')
    ax1.legend()
    
    # Test 2: Distinct eigenvalues
    A, eigs = create_distinct_matrix(n)
    b = np.random.randn(n)
    b /= norm(b)
    x0 = np.zeros(n)
    
    # Run CG
    x_final, history, iters, _ = conjugate_gradient(
        A, b, x0, tolerance=1e-10, max_iterations=1000)
    
    # Compute residuals
    residuals = [norm(b - A @ x) for x in history]
    
    # Plot
    ax2.semilogy(range(len(residuals)), residuals,
                 label='5 Distinct Values', marker='.', markevery=5)
    ax2.grid(True)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Residual Norm')
    ax2.set_title('CG Convergence: 5 Distinct Eigenvalues')
    ax2.legend()
    
    # Save plot
    plt.tight_layout()
    plt.savefig('eigenvalue_clustering.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print eigenvalue information
    print("\nClustered Eigenvalues (with perturbation):")
    A, eigs = create_clustered_matrix(n, perturb=True)
    print(sorted(eigs))
    
    print("\nClustered Eigenvalues (without perturbation):")
    A, eigs = create_clustered_matrix(n, perturb=False)
    print(sorted(eigs))
    
    print("\nDistinct Eigenvalues:")
    A, eigs = create_distinct_matrix(n)
    print(sorted(eigs))

if __name__ == "__main__":
    np.random.seed(42)
    run_eigenvalue_experiments() 