# Tests/compare_cg.py

import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
from scipy.sparse.linalg import cg as scipy_cg
from sklearn.datasets import make_spd_matrix

# 1) Make project root importable (so `models/` is found)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

# 2) Import your new CG
from models.conjugate_gradient import conjugate_gradient

def main() -> None:
    # Problem parameters
    n = 200
    seed = 42
    tol = 1e-8

    # Generate a reproducible SPD matrix and RHS
    A = make_spd_matrix(n_dim=n, random_state=seed)
    rng = np.random.default_rng(seed)
    b = rng.standard_normal(n)
    b /= norm(b)
    x0 = np.zeros(n)

    # --- 1) Run your enhanced CG ---
    start = time.perf_counter()
    x_my, iterates_my, its_my, x_star = conjugate_gradient(
        matrix=A,
        rhs=b,
        initial_guess=x0,
        tolerance=tol,
        max_iterations=n,
        reorthogonalize=False
    )
    time_my = time.perf_counter() - start
    # errors: each row of iterates_my minus the exact solution
    errors_my = iterates_my - x_star.reshape(1, -1)

    # A-norms: sqrt( e^T A e ) for each iterate
    a_norms_my = [
        np.sqrt(err.dot(A.dot(err)))
        for err in errors_my
    ]
    residuals_my = [norm(b - A.dot(xk)) for xk in iterates_my]

    print(f"[My CG]   iterations = {its_my}, time = {time_my:.4f}s, final residual = {residuals_my[-1]:.2e}")
    # (optional) check error vs x_star:
    print(f"[My CG]   error vs exact = {norm(x_my - x_star):.2e}")

    # --- 2) Run SciPy’s CG for comparison ---
    residuals_spy = []
    def record_residual(xk: np.ndarray) -> None:
        residuals_spy.append(norm(b - A.dot(xk)))

    start = time.perf_counter()
    x_spy, info = scipy_cg(
        A,
        b,
        x0=x0,
        rtol=tol,
        maxiter=n,
        callback=record_residual
    )
    time_spy = time.perf_counter() - start

    iters_spy = len(residuals_spy) if info == 0 else info
    print(f"[SciPy CG] info = {info}, iterations = {iters_spy}, time = {time_spy:.4f}s, final residual = {residuals_spy[-1]:.2e}")

    # --- 3) Plot both convergence curves ---
    plt.figure(figsize=(8, 5))
    plt.loglog(range(1, len(residuals_my) + 1), residuals_my,   label="My CG (re‐orthogonalized)", marker='o')
    plt.loglog(range(1, len(residuals_spy) + 1), residuals_spy, label="SciPy CG",              linestyle="--", marker='s')
    plt.xlabel("Iteration")
    plt.ylabel("Residual Norm")
    plt.title("Conjugate Gradient: Custom vs. SciPy")
    plt.legend()
    plt.grid(which="both", ls=":")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8,5))
    plt.loglog(
        range(1, len(a_norms_my)+1),
        a_norms_my,
        label="My CG A-norm error",
        marker='o'
    )
    plt.xlabel("Iteration")
    plt.ylabel(r"$\|x_k - x^*\|_A$")
    plt.title("Conjugate Gradient: Error in $A$-norm")
    plt.legend()
    plt.grid(True, which="both", ls=":")
    plt.show()

if __name__ == "__main__":
    main()
