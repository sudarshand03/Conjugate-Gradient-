import os
import sys
import time

import numpy as np
from numpy.linalg import norm

# 1) Make project root importable
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

# 2) Bring in our modules
from resources.generate_spd import make_spd_matrix
from models.steepest_descent import steepest_descent
from resources.plot_utils import apply_default_style, plot_semilogy, save_plot

def main() -> None:
    # Problem parameters
    n = 5
    seed = 42
    tol = 1e-8
    kappa = 10  # condition number for test SPD matrix

    # 3) Generate test matrix and RHS
    A = make_spd_matrix(n_dim=n, condition_number=kappa, distribution="log", random_state=seed)
    rng = np.random.default_rng(seed)
    b = rng.standard_normal(n)
    b /= norm(b)
    x0 = np.zeros(n)

    # 4) Run Steepest Descent with history
    start = time.perf_counter()
    x_approx, iterates, iterations, x_exact = steepest_descent(
        A, b, x0,
        tolerance=tol,
        max_iterations=1000,
        store_history=True               # record iterates
    )
    elapsed = time.perf_counter() - start
    print(f"Steepest Descent converged in {iterations} iterations ({elapsed:.4f}s)")

    # 5) Plot convergence (2-norm of residual)
    residuals = [norm(b - A.dot(xk)) for xk in iterates]
    apply_default_style()
    plot_semilogy(
        residuals,
        label=f"SD (n={n}, κ={int(kappa)})",
        xlabel="Iteration",
        ylabel="Residual",
        title="Steepest Descent Convergence"
    )
    save_plot(f"sd_convergence_n{n}_kappa{int(kappa)}.png")

if __name__ == "__main__":
    main()