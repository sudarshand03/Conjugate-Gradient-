# Tests/test_custom_cg.py

import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm

# Make sure project root is on the import path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

# Import your modules
from resources.generate_spd import make_spd_matrix
from models.conjugate_gradient import conjugate_gradient

def main() -> None:
    # Problem size & RNG seed
    n = 200
    seed = 12345
    tol = 1e-8

    # -- 1) Build an SPD test matrix (e.g. with condition_number=1000)
    A = make_spd_matrix(
        n_dim=n,
        condition_number=1e3,
        distribution="log",
        random_state=seed
    )

    # RHS and initial guess
    rng = np.random.default_rng(seed)
    b = rng.standard_normal(n)
    b /= norm(b)
    x0 = np.zeros(n)

    # -- 2) Run your enhanced CG solver
    start = time.perf_counter()
    x_approx, iterates, iterations, x_exact = conjugate_gradient(
        matrix=A,
        rhs=b,
        initial_guess=x0,
        tolerance=tol,
        max_iterations=n,
        reorthogonalize=True
    )
    t_cg = time.perf_counter() - start

    # Compute A-norm errors: ||x_k - x*||_A = sqrt((x_k - x*)^T A (x_k - x*))
    errors_A = []
    for xk in iterates:
        e = xk - x_exact
        errors_A.append(np.sqrt(e.dot(A.dot(e))))

    print(f"Custom CG converged in {iterations} iterations in {t_cg:.4f}s")
    print(f"Final A-norm error: {errors_A[-1]:.2e}")

    # -- 3) Plot A-norm error vs iteration
    plt.figure(figsize=(8,5))
    plt.loglog(range(1, len(errors_A)+1), errors_A, marker='o')
    plt.xlabel("Iteration")
    plt.ylabel(r"$\|x_k - x^*\|_A$")
    plt.title("Conjugate Gradient: A-norm Error Convergence")
    plt.grid(which="both", ls=":")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
