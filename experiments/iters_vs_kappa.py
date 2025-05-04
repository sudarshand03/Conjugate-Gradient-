# experiments/iters_vs_kappa.py
"""
Plot iteration counts to reach tolerance vs. condition number for SD and CG.
"""
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm

# 1) Make project root importable
dir_here = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(dir_here, '..'))
sys.path.insert(0, project_root)

# 2) Imports
from resources.generate_spd import generate_spd
from models.steepest_descent import steepest_descent
from models.conjugate_gradient import conjugate_gradient
from resources.plot_utils import apply_default_style

def run_iters_vs_kappa_experiments(
    sizes=(10, 100, 1000),
    cond_nums=np.logspace(1, 4, 20),  # 20 points from 10^1 to 10^4
    tolerance: float = 1e-8,
    max_iter: int = 10000
):
    """
    Analyze iterations vs condition number for different matrix sizes.
    Plots both CG and SD convergence for comparison.
    """
    # ensure results directory
    results_dir = os.path.join(project_root, "results", "cond_nums_iters")
    os.makedirs(results_dir, exist_ok=True)

    # styling
    apply_default_style()

    # Print header for metrics
    print("\nIterations vs Condition Number Analysis:")
    print("-" * 100)
    print(f"{'n':<6} {'κ':<8} {'CG Iters':<10} {'SD Iters':<10} {'CG Time (s)':<12} {'SD Time (s)':<12} {'CG Final Res':<15} {'SD Final Res':<15}")
    print("-" * 100)

    for n in sizes:
        # Create figure for this matrix size
        fig, ax = plt.subplots(figsize=(10, 6))
        
        cg_iters = []
        sd_iters = []
        cg_times = []
        sd_times = []
        cg_residuals = []
        sd_residuals = []

        for kappa in cond_nums:
            # Generate test problem
            A = generate_spd(
                n_dim=n,
                condition_number=float(kappa),
                distribution="log",
                random_state=42
            )
            rng = np.random.default_rng(42)
            b = rng.standard_normal(n)
            b /= norm(b)
            x0 = np.zeros(n)

            # Run CG
            start = time.perf_counter()
            x_cg, _, its_cg, _ = conjugate_gradient(
                matrix=A,
                rhs=b,
                initial_guess=x0,
                tolerance=tolerance,
                max_iterations=max_iter
            )
            cg_time = time.perf_counter() - start
            cg_res = norm(b - A @ x_cg)

            # Run SD
            start = time.perf_counter()
            x_sd, _, its_sd, _ = steepest_descent(
                A, b, x0,
                tolerance=tolerance,
                max_iterations=max_iter,
                store_history=True
            )
            sd_time = time.perf_counter() - start
            sd_res = norm(b - A @ x_sd)

            # Store results
            cg_iters.append(its_cg)
            sd_iters.append(its_sd)
            cg_times.append(cg_time)
            sd_times.append(sd_time)
            cg_residuals.append(cg_res)
            sd_residuals.append(sd_res)

            # Print metrics
            print(f"{n:<6} {kappa:<8.1f} {its_cg:<10} {its_sd:<10} {cg_time:<12.4f} {sd_time:<12.4f} {cg_res:<15.2e} {sd_res:<15.2e}")

        # Plot iterations vs kappa
        ax.loglog(cond_nums, cg_iters, marker='o', label='Conjugate Gradient', linewidth=2)
        ax.loglog(cond_nums, sd_iters, marker='s', label='Steepest Descent', linewidth=2)
        
        ax.set_xlabel('Condition Number κ')
        ax.set_ylabel('Number of Iterations')
        ax.set_title(f'Iterations vs Condition Number (n={n})')
        ax.grid(True, which='both', linestyle='--', alpha=0.7)
        ax.legend()
        fig.tight_layout()

        # Save plot
        out_file = os.path.join(results_dir, f'iters_vs_kappa_n{n}.png')
        fig.savefig(out_file, dpi=300)
        plt.close(fig)
        print(f"\nSaved plot → {out_file}")

        # Create time comparison plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.loglog(cond_nums, cg_times, marker='o', label='Conjugate Gradient', linewidth=2)
        ax.loglog(cond_nums, sd_times, marker='s', label='Steepest Descent', linewidth=2)
        
        ax.set_xlabel('Condition Number κ')
        ax.set_ylabel('Time to Converge (s)')
        ax.set_title(f'Time vs Condition Number (n={n})')
        ax.grid(True, which='both', linestyle='--', alpha=0.7)
        ax.legend()
        fig.tight_layout()

        # Save time plot
        out_file = os.path.join(results_dir, f'time_vs_kappa_n{n}.png')
        fig.savefig(out_file, dpi=300)
        plt.close(fig)
        print(f"Saved time plot → {out_file}")

if __name__ == "__main__":
    np.random.seed(42)
    run_iters_vs_kappa_experiments(
        sizes=(10, 100, 1000),
        cond_nums=np.logspace(1, 4, 20),  # 20 points from 10^1 to 10^4
        tolerance=1e-8,
        max_iter=10000
    )
