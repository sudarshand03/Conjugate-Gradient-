# from typing import Tuple, Optional
# import numpy as np

# def steepest_descent(A: np.ndarray, b: np.ndarray, x0: np.ndarray, tolerance: float = 1e-8, 
#                      max_iterations: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, int, np.ndarray]:
#     """
#     Preconditioned Steepest Descent method for solving Ax = b, where A is SPD.

#     Args:
#         A (np.ndarray): SPD matrix of shape (n, n).
#         b (np.ndarray): Right-hand side vector of shape (n,).
#         x0 (np.ndarray): Initial guess vector of shape (n,).
#         tolerance (float): Convergence threshold for residual norm ||r||, default 1e-8.
#         max_iterations (Optional[int]): Maximum iterations; defaults to dynamic value.

#     Returns:
#         Tuple[np.ndarray, np.ndarray, int, np.ndarray]:
#             - x: Approximate solution.
#             - iterates: Array of all iterates, shape (k+1, n).
#             - k: Number of iterations performed.
#             - x_star: True solution from np.linalg.solve.

#     Raises:
#         ValueError: If A is not SPD or dimensions mismatch.
#     """
#     n = len(b)
#     # Validate input dimensions
#     if A.shape != (n, n) or x0.shape != (n,):
#         raise ValueError("Dimensions of A, b, and x0 must align: A (n,n), b (n,), x0 (n,).")

#     # Check if A is symmetric positive definite
#     if not np.allclose(A, A.T, atol=1e-10):
#         raise ValueError("Matrix A must be symmetric for Steepest Descent.")
#     if np.any(np.linalg.eigvalsh(A) <= 0):
#         raise ValueError("Matrix A must be positive definite for Steepest Descent.")

#     # Estimate condition number to set dynamic max iterations
#     eig_vals = np.linalg.eigvalsh(A)
#     kappa_est = max(eig_vals) / min(eig_vals) if min(eig_vals) > 0 else 1e6
#     max_iter = max_iterations if max_iterations is not None else min(max(2 * n, 50000), 
#                                                                      int(2 * kappa_est * np.log(1 / tolerance)))

#     # Compute true solution for reference
#     x_star = np.linalg.solve(A, b)

#     # Diagonal preconditioning
#     M_inv = np.diag(1.0 / np.diag(A))
#     x = x0.copy()
#     r = b - A @ x  # Initial residual
#     z = M_inv @ r  # Preconditioned residual
#     iterates = np.zeros((max_iter + 1, n))
#     iterates[0, :] = x

#     k = 0
#     r_prev = None
#     while k < max_iter:
#         # Check convergence
#         if np.linalg.norm(r) < tolerance:
#             return x, iterates[:k + 1, :], k + 1, x_star

#         # Compute step size
#         rTz = r @ z
#         Az = A @ z
#         alpha = rTz / (z @ Az)
#         if alpha <= 0 or np.isnan(alpha):
#             alpha = 1e-6  # Fallback for numerical stability

#         # Update solution
#         x = x + alpha * z
#         iterates[k + 1, :] = x
#         r_new = b - A @ x  # Direct residual computation

#         # Log inner product of consecutive residuals
#         if k > 0:
#             r_inner = r_prev @ r_new
#             print(f"Iter {k}: Inner product <r_{k-1}, r_{k}> = {r_inner:.6e}")

#         # Update preconditioned residual
#         z = M_inv @ r_new
#         r_prev = r.copy()
#         r = r_new
#         k += 1

#     return x, iterates[:k + 1, :], k, x_star

# def plot_residual_diagram(A, b, iterates, filename="residual_diagram_n2.png"):
#     """
#     Plot residual vectors for the first few iterations in 2D.

#     Args:
#         A (np.ndarray): SPD matrix of shape (2, 2).
#         b (np.ndarray): Right-hand side vector of shape (2,).
#         iterates (np.ndarray): Array of iterates from Steepest Descent.
#         filename (str): Output file path for the plot.
#     """
#     plt.figure(figsize=(8, 6))
#     # Plot up to first 3 residuals
#     for k in range(min(3, len(iterates))):
#         r_k = b - A @ iterates[k]
#         plt.quiver(0, 0, r_k[0], r_k[1], angles='xy', scale_units='xy', scale=1, 
#                    label=f'r_{k}', color=f'C{k}')
#     plt.xlabel('x1')
#     plt.ylabel('x2')
#     plt.title('Residual Vectors in Steepest Descent (n=2)')
#     plt.legend()
#     plt.grid(True)
#     plt.savefig(filename, dpi=300)
#     plt.close()


import logging
from typing import Callable, Optional, Tuple, Union
import numpy as np
from numpy import ndarray
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt


Logger = logging.getLogger(__name__)

import logging
from typing import Optional, Tuple

import numpy as np
from numpy import ndarray

Logger = logging.getLogger(__name__)


def steepest_descent(
    A: ndarray,
    b: ndarray,
    initial_guess: ndarray,
    tolerance: float = 1e-8,
    max_iterations: Optional[int] = None,
    store_history: bool = True
) -> Tuple[ndarray, Optional[ndarray], int, ndarray]:
    """
    Steepest Descent method for solving Ax = b (non-preconditioned).

    Parameters
    ----------
    A : ndarray, shape (n, n)
        Symmetric positive-definite matrix.
    b : ndarray, shape (n,)
        Right-hand side vector.
    initial_guess : ndarray, shape (n,)
        Initial solution estimate.
    tolerance : float, default=1e-8
        Convergence tolerance for the residual norm.
    max_iterations : int, optional
        Maximum number of iterations (defaults to n).
    store_history : bool, default=True
        Whether to store and return the solution history.

    Returns
    -------
    x : ndarray, shape (n,)
        Approximate solution.
    history : ndarray or None
        Array of iterates (k+1, n) if store_history else None.
    iterations : int
        Number of iterations performed.
    x_star : ndarray, shape (n,)
        Exact solution for reference.
    """
    # Validate shapes
    n = b.shape[0]
    if A.shape != (n, n) or initial_guess.shape != (n,):
        raise ValueError("Shape mismatch: A must be (n,n) and initial_guess (n,)")

    # Symmetry and definiteness check
    if not np.allclose(A, A.T, atol=1e-10):
        raise ValueError("Matrix A must be symmetric.")
    eigs = np.linalg.eigvalsh(A)
    if eigs.min() <= 0:
        raise ValueError("Matrix A must be positive-definite.")

    # Initialize solution and compute exact solution
    x = initial_guess.copy()
    x_star = np.linalg.solve(A, b)

    # Initial residual
    r = b - A @ x
    resid_norm = float(np.linalg.norm(r))

    # History storage
    if max_iterations is None:
        max_iterations = n
    history = np.zeros((max_iterations + 1, n), dtype=A.dtype) if store_history else None
    if store_history:
        history[0] = x

    iteration = 0
    while iteration < max_iterations and resid_norm > tolerance:
        # Compute step length alpha = (r^T r) / (r^T A r)
        Ar = A @ r
        rr = float(r.dot(r))
        denom = float(r.dot(Ar))
        if denom <= 0 or np.isnan(denom):
            Logger.warning("Non-positive denom at iter %d: %f", iteration, denom)
            break
        alpha = rr / denom

        # Update solution and residual by recurrence
        x += alpha * r
        r -= alpha * Ar
        resid_norm = float(np.linalg.norm(r))

        iteration += 1
        if store_history:
            history[iteration] = x

        Logger.debug("Iter %d: resid_norm=%.3e", iteration, resid_norm)

    # Return final solution
    if store_history:
        return x, history[: iteration + 1], iteration, x_star
    return x, None, iteration, x_star



def compute_residual_history(
    A: ndarray,
    b: ndarray,
    history: ndarray
) -> ndarray:
    """
    Given the solution iterates history, compute the corresponding residuals r_k = b - A x_k.

    Returns
    -------
    residuals : ndarray, shape (len(history), n)
        Residual vectors at each iteration.
    """
    return np.array([b - A.dot(xk) for xk in history])


def compute_inner_products(
    residuals: ndarray
) -> np.ndarray:
    """
    Compute inner products <r_{k-1}, r_k> for consecutive residual vectors.

    Returns
    -------
    ips : ndarray, shape (len(residuals)-1,)
        Inner products between consecutive residuals.
    """
    return np.array([
        float(residuals[k-1].dot(residuals[k]))
        for k in range(1, len(residuals))
    ])


def verify_orthogonality_and_plot(
    A: ndarray,
    b: ndarray,
    history: ndarray,
    max_arrows: int = 3,
    filename: Optional[str] = None
) -> None:
    """
    Verify that consecutive residuals are orthogonal (<r_{k-1}, r_k> ≈ 0) and
    plot the first few residual vectors in 2D for intuition.

    Parameters
    ----------
    A : ndarray, shape (n, n)
        SPD matrix used in SD.
    b : ndarray, shape (n,)
        Right-hand side vector.
    history : ndarray, shape (k+1, n)
        Solution iterates from steepest_descent.
    max_arrows : int, default=3
        Number of residual vectors to plot.
    filename : str, optional
        If given, save the plot to this filename.
    """
    # Compute residuals and inner products
    residuals = compute_residual_history(A, b, history)
    ips = compute_inner_products(residuals)

    # Print inner products
    for i, ip in enumerate(ips):
        print(f"<r_{i}, r_{i+1}> = {ip:.3e}")

    # Plot quiver of residuals
    plt.figure(figsize=(6, 6))
    for k, r in enumerate(residuals[:max_arrows]):
        plt.quiver(
            0,
            0,
            r[0],
            r[1],
            angles='xy',
            scale_units='xy',
            scale=1,
            label=f'r_{k}'
        )
    plt.axhline(0, color='gray', linewidth=1)
    plt.axvline(0, color='gray', linewidth=1)
    plt.legend()
    plt.xlabel('x₁')
    plt.ylabel('x₂')
    plt.title('Steepest Descent Residuals (Orthogonality)')
    plt.axis('equal')
    plt.grid(True)
    if filename:
        plt.savefig(filename, dpi=300)
    plt.show()


