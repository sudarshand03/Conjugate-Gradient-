from typing import Tuple, Optional
import numpy as np

def steepest_descent(A: np.ndarray, b: np.ndarray, x0: np.ndarray, tolerance: float = 1e-8, 
                     max_iterations: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    """
    Steepest Descent method for solving Ax = b with SPD A, per Saptoka's pseudocode.

    Args:
        A (np.ndarray): SPD matrix (n x n).
        b (np.ndarray): Right-hand side vector (n).
        x0 (np.ndarray): Initial guess vector (n).
        tolerance (float): Residual norm convergence threshold (default: 1e-8).
        max_iterations (Optional[int]): Maximum iterations (defaults to n).

    Returns:
        Tuple[np.ndarray, np.ndarray, int, np.ndarray]: Solution x, iterates, iterations, true solution x_star.
    """
    # Check input consistency
    n: int = len(b)
    if A.shape != (n, n) or x0.shape != (n,):
        raise ValueError("A, b, and x0 dimensions must align.")

    # Set iteration limit
    max_iter: int = max_iterations if max_iterations is not None else n

    # Compute true solution (Ryan's feature)
    x_star: np.ndarray = np.linalg.solve(A, b)

    # Initialize variables
    x: np.ndarray = x0.copy()           # Current solution
    r: np.ndarray = b - A @ x           # Initial residual
    iterates: np.ndarray = np.zeros((max_iter + 1, n))  # Track all iterates (Ryan's approach)
    iterates[0, :] = x

    # Iteration loop
    k: int = 0
    while k < max_iter:
        # Check residual convergence (Saptoka's standard)
        if np.linalg.norm(r) < tolerance:
            return x, iterates[:k + 1, :], k + 1, x_star

        # Step size computation
        rTr: float = r @ r              # Residual inner product
        Ar: np.ndarray = A @ r          # Matrix-vector product
        alpha: float = rTr / (r @ Ar)   # Optimal step length

        # Update solution and store iterate
        x = x + alpha * r
        iterates[k + 1, :] = x

        # Update residual
        r = r - alpha * Ar
        k += 1

    # Max iterations reached
    return x, iterates[:k + 1, :], k, x_star