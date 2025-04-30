from typing import Tuple, Optional
from models.steepest_descent import steepest_descent
from models.conjugate_gradient import conjugate_gradient
import numpy as np
from scipy.linalg import *

def build_mvot_system(Sigma: np.ndarray, mu: np.ndarray, mu_p: float, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    MVOT System builder method which constructs the MVOT linear system in matrix form.

    Args:
        Sigma (np.ndarray): Covariance matrix of asset returns (n x n), assumed to be symmetric and positive definite.
        mu (np.ndarray): Expected return vector for each asset (length n).
        mu_p (float): Target expected return for the portfolio.
        n (int): number of assets

    Returns:
        Tuple[np.ndarray, np.ndarray, float]:
            - A_reduced (np.ndarray): SPD reduced null-space system
            - b_reduced (np.ndarray): The right-hand side vector.
    """
    # Build constraint matrix
    A_con = np.vstack([mu, np.ones(n)])
    b_con = np.array([mu_p, 1.0])

    # Feasible point x0
    x0, *_ = np.linalg.lstsq(A_con, b_con, rcond=None)

    # Nullspace basis Z
    Z = null_space(A_con)

    # Build reduced SPD system
    A_reduced = Z.T @ Sigma @ Z
    b_reduced = -Z.T @ Sigma @ x0

    # Initial guess
    y0 = np.zeros_like(b_reduced)

    return A_reduced, b_reduced, x0, y0, Z

# SD solver wrapper for MVOT
def solve_mvot_sd(A, b, tol=1e-8, max_iter=10000):
    x0 = np.zeros(len(b))
    return steepest_descent(A,b,x0,tol,max_iter)

# CG solver wrapper for MVOT
def solve_mvot_cg(A, b, tol=1e-8, max_iter=10000):
    x0 = np.zeros(len(b))
    return conjugate_gradient(A,b,x0,tol,max_iter)
