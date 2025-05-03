import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm, eigh
from models.conjugate_gradient import conjugate_gradient
from models.steepest_descent import steepest_descent

def create_clustered_spd(n=10, clusters=5):
    """Create SPD matrix with eigenvalues in 5 clusters."""
    # Create clustered eigenvalues
    cluster_centers = [1, 10, 100, 1000, 10000]  # 5 clusters
    eigenvalues = []
    for center in cluster_centers:
        # Add 2 eigenvalues per cluster with small perturbation
        eigenvalues.extend([center * (1 + 0.1 * np.random.randn()) for _ in range(2)])
    
    # Generate random orthogonal matrix
    Q = np.random.randn(n, n)
    Q, _ = np.linalg.qr(Q)
    
    # Create diagonal matrix of eigenvalues
    D = np.diag(eigenvalues)
    
    # Create SPD matrix A = QDQ^T
    A = Q @ D @ Q.T
    
    return A, eigenvalues

def create_distinct_spd(n=10):
    """Create SPD matrix with exactly 5 distinct eigenvalues."""
    # Exactly 5 distinct eigenvalues
    eigenvalues = [1, 10, 100, 1000, 10000]
    eigenvalues = eigenvalues * 2  # Repeat each twice to get n=10
    
    # Generate random orthogonal matrix
    Q = np.random.randn(n, n)
    Q, _ = np.linalg.qr(Q)
    
    # Create diagonal matrix of eigenvalues
    D = np.diag(eigenvalues)
    
    # Create SPD matrix A = QDQ^T
    A = Q @ D @ Q.T
    
    return A, eigenvalues

def create_nonsymmetric_spd(n=10):
    """Create non-symmetric matrix and its normal equations matrix."""
    # Create random non-symmetric matrix
    M = np.random.randn(n, n)
    
    # Form normal equations matrix A = M^T M (always SPD)
    A = M.T @ M
    
    # Get eigenvalues of A
    eigenvalues = np.linalg.eigvals(A)
    
    return A, M, eigenvalues

def run_experiment(A, b, x0, title, max_iter=100):
    """Run CG and SD on given system and plot convergence."""
    # Run CG
    x_cg, history_cg, iters_cg, _ = conjugate_gradient(
        A, b, x0, tolerance=1e-10, max_iterations=max_iter)
    
    # Run SD
    x_sd, history_sd, iters_sd, _ = steepest_descent(
        A, b, x0, tolerance=1e-10, max_iterations=max_iter)
    
    # Compute residual norms
    residuals_cg = [norm(b - A @ x) for x in history_cg]
    residuals_sd = [norm(b - A @ x) for x in history_sd]
    
    # Plot convergence
    plt.figure(figsize=(10, 6))
    plt.semilogy(range(len(residuals_cg)), residuals_cg, 
                 'b-', label='CG', marker='.', markevery=5)
    plt.semilogy(range(len(residuals_sd)), residuals_sd, 
                 'r-', label='SD', marker='.', markevery=5)
    plt.grid(True)
    plt.xlabel('Iteration')
    plt.ylabel('Residual Norm')
    plt.title(f'Convergence: {title}')
    plt.legend()
    
    # Save plot
    filename = title.lower().replace(' ', '_') + '.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    return iters_cg, iters_sd

def main():
    np.random.seed(42)
    n = 10  # Matrix size
    
    # Test 1: 5-cluster SPD matrix
    A_cluster, eigs_cluster = create_clustered_spd(n)
    b_cluster = np.random.randn(n)
    x0_cluster = np.zeros(n)
    print("\n5-Cluster Matrix Eigenvalues:", sorted(eigs_cluster))
    iters_cluster_cg, iters_cluster_sd = run_experiment(
        A_cluster, b_cluster, x0_cluster, "5-Cluster SPD Matrix")
    
    # Test 2: 5 distinct eigenvalues
    A_distinct, eigs_distinct = create_distinct_spd(n)
    b_distinct = np.random.randn(n)
    x0_distinct = np.zeros(n)
    print("\n5-Distinct Matrix Eigenvalues:", sorted(eigs_distinct))
    iters_distinct_cg, iters_distinct_sd = run_experiment(
        A_distinct, b_distinct, x0_distinct, "5 Distinct Eigenvalues")
    
    # Test 3: Non-symmetric case (using normal equations)
    A_nonsym, M, eigs_nonsym = create_nonsymmetric_spd(n)
    b_nonsym = np.random.randn(n)
    x0_nonsym = np.zeros(n)
    print("\nNon-symmetric Matrix Eigenvalues:", sorted(np.real(eigs_nonsym)))
    iters_nonsym_cg, iters_nonsym_sd = run_experiment(
        A_nonsym, b_nonsym, x0_nonsym, "Non-symmetric Case (Normal Equations)")

if __name__ == "__main__":
    main() 