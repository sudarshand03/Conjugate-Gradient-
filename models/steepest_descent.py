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
    max_iterations: int = 1000,
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
    max_iterations : int, default=1000
        Maximum number of iterations to perform.
    store_history : bool, default=True
        Whether to store and return the full iterate history.

    Returns
    -------
    x : ndarray, shape (n,)
        Final approximate solution.
    history : ndarray or None
        Array of iterates (shape (k+1, n)) if store_history else None.
    iterations : int
        Number of iterations actually performed.
    x_star : ndarray, shape (n,)
        Exact solution computed via np.linalg.solve.
    """
    # 1) Validate shapes
    n = b.shape[0]
    if A.shape != (n, n) or initial_guess.shape != (n,):
        raise ValueError("Shape mismatch: A must be (n,n) and initial_guess (n,)")

    # 2) Symmetry & positive-definite check
    if not np.allclose(A, A.T, atol=1e-10):
        raise ValueError("Matrix A must be symmetric.")
    eigs = np.linalg.eigvalsh(A)
    if eigs.min() <= 0:
        raise ValueError("Matrix A must be positive-definite.")

    # 3) Initialize solution and exact reference
    x = initial_guess.copy()
    x_star = np.linalg.solve(A, b)

    # 4) Compute initial residual
    r = b - A @ x
    resid_norm = float(np.linalg.norm(r))

    # 5) Allocate history array (honoring max_iterations)
    history = np.zeros((max_iterations + 1, n), dtype=A.dtype) if store_history else None
    if store_history:
        history[0] = x

    # 6) Main loop: run until tolerance or until we exhaust max_iterations
    iteration = 0
    while iteration < max_iterations and resid_norm > tolerance:
        Ar = A @ r
        rr = float(r.dot(r))
        denom = float(r.dot(Ar))
        if denom <= 0 or np.isnan(denom):
            Logger.warning("Non-positive denom at iter %d: %f", iteration, denom)
            break
        alpha = rr / denom

        # update x, residual, norm
        x += alpha * r
        r -= alpha * Ar
        resid_norm = float(np.linalg.norm(r))

        iteration += 1
        if store_history:
            history[iteration] = x

        Logger.debug("Iter %d: resid_norm=%.3e", iteration, resid_norm)

    # 7) Return results (slice history to actual iterations+1)
    if store_history:
        return x, history[: iteration + 1], iteration, x_star
    else:
        return x, None, iteration, x_star
    
