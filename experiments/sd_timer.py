# experiments/sd_timer.py

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
from resources.plot_utils import apply_default_style


def run_sd_timer(
    sizes=(10, 100, 1000),
    cond_nums=(10, 100, 1000),
    tolerance: float = 1e-8,
    max_iter: int = 5000
):
    """
    Measure and plot SD time-to-converge vs matrix size for several κ.

    Args:
        sizes: tuple of matrix dimensions to test.
        cond_nums: tuple of condition numbers κ to test.
        tolerance: convergence tolerance on the residual norm.
        max_iter: hard cap on SD iterations.
    """
    # ensure results directory
    results_dir = os.path.abspath("results")
    os.makedirs(results_dir, exist_ok=True)

    # styling
    apply_default_style()

    # prepare the plot
    fig, ax = plt.subplots(figsize=(8,6))

    for kappa in cond_nums:
        times = []
        for n in sizes:
            # build test problem
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

            # time the SD run (no history)
            start = time.perf_counter()
            _, _, its, _ = steepest_descent(
                A, b, x0,
                tolerance=tolerance,
                max_iterations=max_iter,
                store_history=False
            )
            elapsed = time.perf_counter() - start
            times.append(elapsed)

            print(f"κ={kappa:4d}, n={n:5d}, its={its:4d}, time={elapsed:.3e}s")

        # plot this κ’s curve
        ax.loglog(
            sizes,
            times,
            marker='o',
            linewidth=2,
            label=f'κ={kappa}'
        )

    # finalize plot
    ax.set_xlabel("Matrix size $n$")
    ax.set_ylabel("Time to converge (s)")
    ax.set_title("Steepest Descent Time vs Matrix Size")
    ax.grid(True, which='both', linestyle='--', alpha=0.7)
    ax.legend(title="Condition Number")
    fig.tight_layout()

    # save
    out_file = os.path.join(results_dir, "sd_time_vs_size_multiple_kappa.png")
    fig.savefig(out_file, dpi=300)
    plt.close(fig)

    print(f"Saved timing plot → {out_file}")


if __name__ == "__main__":
    run_sd_timer(
        sizes=(10, 100, 1000),
        cond_nums=(10, 100, 1000),
        tolerance=1e-8,
        max_iter=5000
    )
