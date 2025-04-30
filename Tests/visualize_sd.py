# import os
# import sys
# import numpy as np
# from numpy.linalg import norm

# from typing import Optional

# # Ensure project root is on sys.path
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# sys.path.insert(0, project_root)

# from models.steepest_descent import steepest_descent
# from resources.generate_spd import make_spd_matrix
# from resources.plot_utils import plot_semilogy, save_plot


# def visualize_steepest_descent(
#     n: int = 100,
#     condition_number: float = 100.0,
#     tol: float = 1e-8,
#     max_iter: Optional[int] = None
# ) -> None:
#     """
#     Run Steepest Descent on a random SPD matrix and plot residual norms vs iteration.

#     Parameters
#     ----------
#     n : int
#         Dimension of the SPD matrix.
#     condition_number : float
#         Desired condition number for the SPD matrix.
#     tol : float
#         Convergence tolerance for the residual norm.
#     max_iter : int, optional
#         Maximum iterations (defaults to n).
#     """
#     # Generate test problem
#     A = make_spd_matrix(
#         n_dim=n,
#         condition_number=condition_number,
#         distribution='linear',
#         random_state=42
#     )
#     rng = np.random.default_rng(42)
#     b = rng.standard_normal(n)
#     b /= norm(b)
#     x0 = np.zeros(n)

#     # Solve via Steepest Descent (non-preconditioned)
#     x_sd, history, its, x_star = steepest_descent(
#         A,
#         b,
#         x0,
#         tolerance=tol,
#         max_iterations=max_iter,
#         store_history=True
#     )

#     # Compute residual norms
#     residuals = [norm(b - A.dot(xk)) for xk in history]

#     # Plot using utilities
#     plot_semilogy(
#         y_values=residuals,
#         label='Steepest Descent',
#         xlabel='Iteration',
#         ylabel=r'$\|b - A x_k\|$',
#         title=f'Steepest Descent Convergence (n={n}, κ={condition_number})'
#     )
#     # Save plot into results/
#     filename = f'sd_convergence_n{n}_kappa{int(condition_number)}.png'
#     save_plot(filename)


# if __name__ == '__main__':
#     visualize_steepest_descent(n=100, condition_number=100.0)

# Tests/visualize_sd.py

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
from models.steepest_descent import steepest_descent, compute_residual_history, verify_orthogonality_and_plot
from resources.plot_utils import apply_default_style, plot_semilogy, save_plot

def main() -> None:
    # Problem parameters
    n = 200
    seed = 42
    tol = 1e-8
    kappa = 1e3  # condition number for test SPD matrix

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
        max_iterations=n,
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
        ylabel="Residual ‖b – A xₖ‖₂",
        title="Steepest Descent Convergence"
    )
    save_plot(f"sd_convergence_n{n}_kappa{int(kappa)}.png")

    # 6) Verify orthogonality and plot residual vectors
    #    This will print ⟨r_{k-1},r_k⟩ to console and save a quiver plot
    verify_orthogonality_and_plot(
        A,
        b,
        iterates,
        filename=f"sd_orthogonality_n{n}.png"
    )

if __name__ == "__main__":
    main()