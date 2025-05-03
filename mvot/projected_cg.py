import numpy as np

def projected_conjugate_gradient(P, c, A, b, tol=1e-8, maxiter=None):
    """
    Solve:
        minimize    xᵀ P x  – bᵀ x
        subject to  A x = c

    using Projected Conjugate Gradient.

    Parameters
    ----------
    P : (n, n) ndarray
        SPD matrix in the quadratic term.
    c : (m,) ndarray
        Right‐hand side of the m linear constraints.
    A : (m, n) ndarray
        Constraint matrix, with full row rank m < n.
    b : (n,) ndarray
        Linear term in the objective.
    tol : float
        Tolerance for stopping on the projected residual.
    maxiter : int or None
        Maximum iterations (defaults to n−m).

    Returns
    -------
    x : (n,) ndarray
        The optimal solution satisfying A x = c.
    """
    n = P.shape[0]
    m = A.shape[0]
    if maxiter is None:
        maxiter = n - m

    # Precompute projection onto null(A): P_c = I - Aᵀ (A Aᵀ)⁻¹ A
    M = A @ A.T
    M_inv = np.linalg.inv(M)
    P_c = np.eye(n) - A.T @ (M_inv @ A)

    # Feasible starting point: x₀ = Aᵀ (A Aᵀ)⁻¹ c
    x = A.T @ (M_inv @ c)

    # Initial (projected) residual
    r = P @ x - b
    Rc = P_c @ r
    p = -Rc.copy()
    rsold = Rc.dot(Rc)

    for k in range(maxiter):
        Ap = P @ p
        alpha = rsold / p.dot(Ap)
        x += alpha * p
        r = P @ x - b
        Rc = P_c @ r
        rsnew = Rc.dot(Rc)
        if np.sqrt(rsnew) < tol:
            break
        beta = rsnew / rsold
        p = -Rc + beta * p
        rsold = rsnew

    return x
