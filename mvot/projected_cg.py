import numpy as np
from typing import Optional, Union


def projected_conjugate_gradient(
    P: np.ndarray,
    c: np.ndarray,
    A: np.ndarray,
    b: Optional[np.ndarray] = None,
    tol: float = 1e-8,
    maxiter: Optional[int] = None
) -> np.ndarray:
    """
    Solve the equality‐constrained QP via projected conjugate gradient.

    Parameters
    ----------
    P : (n, n) array_like
        SPD matrix (e.g. covariance).
    c : (m,) array_like
        RHS of the m linear constraints.
    A : (m, n) array_like
        Constraint matrix, full-row-rank.
    b : (n,) array_like, optional
        Linear term (default: zero vector).
    tol : float, optional
        Tolerance for stopping on the projected residual.
    maxiter : int, optional
        Maximum iterations (default: n - m).

    Returns
    -------
    x : (n,) ndarray
        The solution satisfying A x = c.
    """
    P = np.asarray(P)
    A = np.asarray(A)
    c = np.asarray(c)
    n, m = P.shape[0], A.shape[0]

    if b is None:
        b = np.zeros(n, dtype=P.dtype)
    else:
        b = np.asarray(b)

    if maxiter is None:
        maxiter = 500000

    # Build projector onto ker(A):  P_c = I - Aᵀ (A Aᵀ)⁻¹ A
    M = A @ A.T
    # solve M Y = A  ⇒  Y = M⁻¹ A
    Y = np.linalg.solve(M, A)
    P_c = np.eye(n) - A.T @ Y

    # Feasible starting point: x₀ = Aᵀ (A Aᵀ)⁻¹ c
    x = A.T @ np.linalg.solve(M, c)

    # Initial projected residual
    residual = P @ x - b
    proj_residual = P_c @ residual
    search_dir = -proj_residual
    rs_old = proj_residual @ proj_residual

    for _ in range(maxiter):
        Pp = P @ search_dir
        alpha = rs_old / (search_dir @ Pp)
        x += alpha * search_dir

        residual = P @ x - b
        proj_residual = P_c @ residual
        rs_new = proj_residual @ proj_residual
        if rs_new < tol**2:
            break

        beta = rs_new / rs_old
        search_dir = -proj_residual + beta * search_dir
        rs_old = rs_new

    return x
