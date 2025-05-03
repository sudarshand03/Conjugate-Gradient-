import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
from models.conjugate_gradient import conjugate_gradient

def create_nonsymmetric_system(n: int, kappa: float):
    """Create a non-symmetric system and its normal equations."""
    # Create random non-symmetric matrix with desired condition number
    U, _ = np.linalg.qr(np.random.randn(n, n))
    V, _ = np.linalg.qr(np.random.randn(n, n))
    s = np.logspace(0, np.log10(kappa), n)
    M = U @ np.diag(s) @ V.T
    
    # Form normal equations
    A = M.T @ M  # This is SPD
    
    return M, A

def run_nonsymmetric_experiment():
    """Compare CG convergence on normal equations vs original SPD systems."""
    sizes = [10, 100]
    kappas = [10, 100, 1000]
    
    for n in sizes:
        plt.figure(figsize=(12, 6))
        
        for kappa in kappas:
            # Non-symmetric case
            M, A = create_nonsymmetric_system(n, kappa)
            b = np.random.randn(n)
            b /= norm(b)
            x0 = np.zeros(n)
            
            # Solve normal equations with CG
            x_final, history, iters, _ = conjugate_gradient(
                A, M.T @ b, x0, tolerance=1e-10, max_iterations=1000)
            
            # Compute residuals for original system
            residuals = [norm(b - M @ x) for x in history]
            
            plt.semilogy(range(len(residuals)), residuals,
                        label=f'κ={kappa}', marker='.', markevery=5)
        
        plt.grid(True)
        plt.xlabel('Iteration')
        plt.ylabel('Residual Norm')
        plt.title(f'CG Convergence on Normal Equations (n={n})')
        plt.legend()
        plt.savefig(f'nonsymmetric_convergence_n{n}.png', dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    np.random.seed(42)
    run_nonsymmetric_experiment() 