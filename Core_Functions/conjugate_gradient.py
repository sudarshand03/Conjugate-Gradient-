from typing import Tuple, Optional
import numpy as np

def conjugate_gradient(A: np.ndarray, b: np.ndarray, x0: np.ndarray, tolerance: float = 1e-8, 
                      max_iterations: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    """
    Conjugate Gradient method for solving Ax = b with SPD A, per Saptoka's pseudocode.

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

    # Compute true solution
    x_star: np.ndarray = np.linalg.solve(A, b)

    # Initialize variables
    x: np.ndarray = x0.copy()
    r: np.ndarray = b - A @ x
    p: np.ndarray = r.copy()
    iterates: np.ndarray = np.zeros((max_iter + 1, n))
    iterates[0, :] = x
    rTr_old: float = r @ r

    # Iteration loop
    k: int = 0
    while k < max_iter:
        if np.linalg.norm(r) < tolerance:
            return x, iterates[:k + 1, :], k + 1, x_star

        # Step size computation with numerical stability
        Ap: np.ndarray = A @ p
        pAp: float = p @ Ap
        if pAp <= 0:
            raise ValueError("A not positive-definite.")
        alpha: float = rTr_old / pAp

        # Update solution and store iterate
        x = x + alpha * p
        iterates[k + 1, :] = x

        # Update residual
        r_new: np.ndarray = r - alpha * Ap
        rTr_new: float = r_new @ r_new

        # Compute beta with numerical safeguard
        if rTr_old < 1e-15:
            beta = 0
        else:
            beta = rTr_new / rTr_old

        # Update direction with orthogonalization to previous direction
        p_new = r_new + beta * p
        # Orthogonalize p_new against previous p to reduce numerical error
        if k > 0:
            p_new -= (p_new @ Ap / pAp) * p
        p = p_new

        # Prepare next iteration
        r = r_new
        rTr_old = rTr_new
        k += 1

    return x, iterates[:k + 1, :], k, x_star