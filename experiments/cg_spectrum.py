# experiments/cg_spectrum_comparison.py

import os
import sys

# 1) Make project root importable
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm

# 2) Import your modules
from resources.generate_spd import make_spd_matrix
from models.conjugate_gradient import conjugate_gradient
from resources.plot_utils import apply_default_style

def cg_l2_residuals(A, b, tol=1e-10, maxiter=None):
    """
    Run your conjugate_gradient solver and return the L2 residual norms
    at each iterate.
    """
    x_cg, iterates, its, x_star = conjugate_gradient(
        matrix=A,
        rhs=b,
        initial_guess=np.zeros_like(b),
        tolerance=tol,
        max_iterations=maxiter
    )
    return [ norm(b - A.dot(xk)) for xk in iterates ]

def main():
    # PARAMETERS — tweak as you like
    n = 200                # matrix dimension
    kappa = 100            # “spread” of the spectrum
    tol = 1e-10
    maxiter = 500

    # Ensure results dir
    results_dir = os.path.abspath("results")
    os.makedirs(results_dir, exist_ok=True)

    # Build a random orthonormal basis Q via your SPD generator
    # (we only use make_spd_matrix to get a random Q; eigenvalues come from us)
    # To force use of our eigenvalues, we pass them explicitly:
    rng = np.random.default_rng(42)
    # 3) Distinct spectrum: linearly spaced from 1 → kappa
    eig_distinct = np.linspace(1.0, kappa, n)
    A_distinct = make_spd_matrix(n_dim=n, eigenvalues=eig_distinct, random_state=42)

    # 4) Clustered spectrum: half at 1, half at kappa
    eig_clustered = np.concatenate([
        np.ones(n//2),
        kappa * np.ones(n - n//2)
    ])
    A_clustered = make_spd_matrix(n_dim=n, eigenvalues=eig_clustered, random_state=42)

    # 5) Fixed RHS
    b = rng.standard_normal(n)
    b /= norm(b)

    # 6) Compute residual histories
    res_distinct = cg_l2_residuals(A_distinct, b, tol=tol, maxiter=maxiter)
    res_clustered = cg_l2_residuals(A_clustered, b, tol=tol, maxiter=maxiter)

    # 7) Plot
    apply_default_style()
    plt.figure(figsize=(8,5))
    plt.semilogy(res_distinct, marker='o', markevery=10, linewidth=2, label='Distinct spectrum')
    plt.semilogy(res_clustered, marker='s', markevery=10, linewidth=2, label='Clustered spectrum')
    plt.xlabel('Iteration $k$')
    plt.ylabel(r'Residual $\|b - A x_k\|_2$')
    plt.title(f'CG Convergence (n={n}, κ={kappa})')
    plt.grid(which='both', linestyle='--', alpha=0.7)
    plt.legend(loc='upper right')
    plt.tight_layout()

    out_file = os.path.join(results_dir, f'cg_distinct_vs_clustered_n{n}_k{kappa}.png')
    plt.savefig(out_file, dpi=300)
    plt.show()
    print(f"Saved plot to {out_file}")

if __name__ == "__main__":
    main()
