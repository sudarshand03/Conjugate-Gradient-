from typing import Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt

def steepest_descent(A: np.ndarray, b: np.ndarray, x0: np.ndarray, tolerance: float = 1e-8, 
                     max_iterations: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    """
    Preconditioned Steepest Descent method for solving Ax = b, where A is SPD.

    Args:
        A (np.ndarray): SPD matrix of shape (n, n).
        b (np.ndarray): Right-hand side vector of shape (n,).
        x0 (np.ndarray): Initial guess vector of shape (n,).
        tolerance (float): Convergence threshold for residual norm ||r||, default 1e-8.
        max_iterations (Optional[int]): Maximum iterations; defaults to dynamic value.

    Returns:
        Tuple[np.ndarray, np.ndarray, int, np.ndarray]:
            - x: Approximate solution.
            - iterates: Array of all iterates, shape (k+1, n).
            - k: Number of iterations performed.
            - x_star: True solution from np.linalg.solve.

    Raises:
        ValueError: If A is not SPD or dimensions mismatch.
    """
    n = len(b)
    # Validate input dimensions
    if A.shape != (n, n) or x0.shape != (n,):
        raise ValueError("Dimensions of A, b, and x0 must align: A (n,n), b (n,), x0 (n,).")

    # Check if A is symmetric positive definite
    if not np.allclose(A, A.T, atol=1e-10):
        raise ValueError("Matrix A must be symmetric for Steepest Descent.")
    if np.any(np.linalg.eigvalsh(A) <= 0):
        raise ValueError("Matrix A must be positive definite for Steepest Descent.")

    # Estimate condition number to set dynamic max iterations
    eig_vals = np.linalg.eigvalsh(A)
    kappa_est = max(eig_vals) / min(eig_vals) if min(eig_vals) > 0 else 1e6
    max_iter = max_iterations if max_iterations is not None else min(max(2 * n, 50000), 
                                                                     int(2 * kappa_est * np.log(1 / tolerance)))

    # Compute true solution for reference
    x_star = np.linalg.solve(A, b)

    # Diagonal preconditioning
    M_inv = np.diag(1.0 / np.diag(A))
    x = x0.copy()
    r = b - A @ x  # Initial residual
    z = M_inv @ r  # Preconditioned residual
    iterates = np.zeros((max_iter + 1, n))
    iterates[0, :] = x

    k = 0
    r_prev = None
    while k < max_iter:
        # Check convergence
        if np.linalg.norm(r) < tolerance:
            return x, iterates[:k + 1, :], k + 1, x_star

        # Compute step size
        rTz = r @ z
        Az = A @ z
        alpha = rTz / (z @ Az)
        if alpha <= 0 or np.isnan(alpha):
            alpha = 1e-6  # Fallback for numerical stability

        # Update solution
        x = x + alpha * z
        iterates[k + 1, :] = x
        r_new = b - A @ x  # Direct residual computation

        # Log inner product of consecutive residuals
        if k > 0:
            r_inner = r_prev @ r_new
            print(f"Iter {k}: Inner product <r_{k-1}, r_{k}> = {r_inner:.6e}")

        # Update preconditioned residual
        z = M_inv @ r_new
        r_prev = r.copy()
        r = r_new
        k += 1

    return x, iterates[:k + 1, :], k, x_star

def plot_residual_diagram(A, b, iterates, filename="residual_diagram_n2.png"):
    """
    Plot residual vectors for the first few iterations in 2D.

    Args:
        A (np.ndarray): SPD matrix of shape (2, 2).
        b (np.ndarray): Right-hand side vector of shape (2,).
        iterates (np.ndarray): Array of iterates from Steepest Descent.
        filename (str): Output file path for the plot.
    """
    plt.figure(figsize=(8, 6))
    # Plot up to first 3 residuals
    for k in range(min(3, len(iterates))):
        r_k = b - A @ iterates[k]
        plt.quiver(0, 0, r_k[0], r_k[1], angles='xy', scale_units='xy', scale=1, 
                   label=f'r_{k}', color=f'C{k}')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Residual Vectors in Steepest Descent (n=2)')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename, dpi=300)
    plt.close()