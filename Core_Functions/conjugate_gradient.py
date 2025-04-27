from typing import Tuple, Optional
import numpy as np

def conjugate_gradient(A: np.ndarray, b: np.ndarray, x0: np.ndarray, tolerance: float = 1e-8, 
                       max_iterations: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    """
    Conjugate Gradient method for solving Ax = b, where A is symmetric positive definite (SPD).

    Args:
        A (np.ndarray): SPD matrix of shape (n, n).
        b (np.ndarray): Right-hand side vector of shape (n,).
        x0 (np.ndarray): Initial guess vector of shape (n,).
        tolerance (float): Convergence threshold for residual norm ||r||, default 1e-8.
        max_iterations (Optional[int]): Maximum iterations; defaults to n if None.

    Returns:
        Tuple[np.ndarray, np.ndarray, int, np.ndarray]:
            - x: Approximate solution.
            - iterates: Array of all iterates, shape (k+1, n).
            - k: Number of iterations performed.
            - x_star: True solution from np.linalg.solve.

    Raises:
        ValueError: If A is not SPD or dimensions mismatch.
    """
    n: int = len(b)
    # Validate input dimensions
    if A.shape != (n, n) or x0.shape != (n,):
        raise ValueError("Dimensions of A, b, and x0 must align: A (n,n), b (n,), x0 (n,).")

    # Check if A is symmetric positive definite
    if not np.allclose(A, A.T, atol=1e-10):
        raise ValueError("Matrix A must be symmetric for Conjugate Gradient.")
    if np.any(np.linalg.eigvalsh(A) <= 0):
        raise ValueError("Matrix A must be positive definite for Conjugate Gradient.")

    # Set maximum iterations if not provided
    max_iter: int = max_iterations if max_iterations is not None else n
    # Compute true solution for reference
    x_star: np.ndarray = np.linalg.solve(A, b)

    # Initialize variables
    x: np.ndarray = x0.copy()
    r: np.ndarray = b - A @ x  # Initial residual: r_0 = b - A x_0
    p: np.ndarray = r.copy()  # Initial direction: p_0 = r_0
    iterates: np.ndarray = np.zeros((max_iter + 1, n))
    iterates[0, :] = x
    rTr_old: float = r @ r  # Initial residual norm squared

    k: int = 0
    while k < max_iter:
        # Check convergence
        if np.linalg.norm(r) < tolerance:
            return x, iterates[:k + 1, :], k + 1, x_star

        # Compute step size alpha
        Ap: np.ndarray = A @ p
        pAp: float = p @ Ap
        if pAp <= 0:
            raise ValueError("Matrix A is not positive-definite during iteration.")
        alpha: float = rTr_old / pAp  # α_k = (r_k^T r_k) / (p_k^T A p_k)

        # Update solution and residual
        x = x + alpha * p
        iterates[k + 1, :] = x
        r_new: np.ndarray = r - alpha * Ap

        # Compute beta for next direction
        rTr_new: float = r_new @ r_new
        beta: float = rTr_new / rTr_old if rTr_old >= 1e-15 else 0  # Avoid division by very small numbers

        # Update direction with orthogonalization for stability
        p_new = r_new + beta * p
        if k > 0:
            p_new -= (p_new @ Ap / pAp) * p  # Orthogonalize against previous direction
        p = p_new

        r = r_new
        rTr_old = rTr_new
        k += 1

    # Return results if max iterations reached
    return x, iterates[:k + 1, :], k, x_star
