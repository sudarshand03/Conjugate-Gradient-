# experiments/convergence_sd.py
import os
import sys
import time

# 1) Make project root importable
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm

from resources.generate_spd import make_spd_matrix
from models.steepest_descent import steepest_descent

def run_sd_residual_experiments(
    sizes=(10, 100, 1000),
    cond_nums=(10, 100, 1000),
    tol: float = 1e-8,
    max_iter: int = 10000
):
    results_dir = os.path.abspath("results")
    os.makedirs(results_dir, exist_ok=True)

    # Print header for metrics
    print("\nSteepest Descent Convergence Metrics:")
    print("-" * 60)
    print(f"{'n':<6} {'κ':<6} {'Iterations':<12} {'Time (s)':<10} {'Final Residual':<15}")
    print("-" * 60)

    for n in sizes:
        # 2) Create one figure & axis per n
        fig, ax = plt.subplots(figsize=(8, 6))

        for kappa in cond_nums:
            # 3) Build SPD problem
            A = make_spd_matrix(
                n_dim=n,
                condition_number=float(kappa),
                distribution="linear",
                random_state=42
            )
            rng = np.random.default_rng(42)
            b = rng.standard_normal(n)
            b /= norm(b)
            x0 = np.zeros(n)

            # 4) Run SD up to max_iter with timing
            start_time = time.perf_counter()
            x_final, history, its, x_star = steepest_descent(
                A, b, x0,
                tolerance=tol,
                max_iterations=max_iter,
                store_history=True
            )
            elapsed = time.perf_counter() - start_time

            # 5) Compute L2‐residuals at each iterate
            res_norms = [norm(b - A.dot(xk)) for xk in history]
            final_residual = res_norms[-1]

            # Print metrics for this case
            print(f"{n:<6} {kappa:<6} {its:<12} {elapsed:<10.4f} {final_residual:<15.2e}")

            # 6) Plot them all on the same axis
            ax.semilogy(
                np.arange(len(res_norms)),
                res_norms,
                marker=',',
                linestyle='-',
                label=f"κ={kappa}, iters={its}"
            )

        # 7) Tidy up the plot
        ax.set_title(f"SD Residual Convergence (n={n})")
        ax.set_xlabel("Iteration k")
        ax.set_ylabel(r"$\|r_k\|_2 = \|b - A x_k\|_2$")
        ax.grid(True, which="both", linestyle="--", alpha=0.6)
        ax.legend(loc="best")
        fig.tight_layout()

        # 8) Save & close
        out_path = os.path.join(results_dir, f"sd_residuals_n{n}.png")
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"Saved: {out_path}")

if __name__ == "__main__":
    np.random.seed(42)
    run_sd_residual_experiments(
        sizes=(10, 100, 1000),
        cond_nums=(10, 100, 1000),
        tol=1e-8,
        max_iter=10000
    )
