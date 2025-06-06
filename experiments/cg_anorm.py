# experiments/cg_error_convergence.py

import os
import sys
import time

# 1) Make project root importable
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm

from resources.generate_spd import generate_spd
from resources.generate_spd import generate_spd
from models.conjugate_gradient import conjugate_gradient

def run_cg_error_experiments(
    sizes=(10, 100, 1000),
    cond_nums=(10, 100, 1000),
    tolerance: float = 1e-8,
    max_iter: int = None
):
    """
    For each matrix size n in `sizes` and condition number κ in `cond_nums`,
    run Conjugate Gradient until convergence or max_iter, then plot the
    A-norm of the error e_k = x_k - x* versus iteration k.
    """
    results_dir = os.path.join(project_root, "results", "CG_Convergence")
    os.makedirs(results_dir, exist_ok=True)

    # Print header for metrics
    print("\nConjugate Gradient A-Norm Error Metrics:")
    print("-" * 70)
    print(f"{'n':<6} {'κ':<6} {'Iterations':<12} {'Time (s)':<10} {'Final A-Norm Error':<15} {'Time/Iter (ms)':<15}")
    print("-" * 70)

    for n in sizes:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for kappa in cond_nums:
            # Generate test problem
            A = generate_spd(
                n_dim=n, 
                condition_number=kappa, 
                distribution="log",
                random_state=42)
            rng = np.random.default_rng(42)
            b = rng.standard_normal(n)
            b /= norm(b)
            x0 = np.zeros(n)
        
            
            # Run CG with timing
            start_time = time.perf_counter()
            _, history, iters, x_star = conjugate_gradient(matrix=A, rhs=b, initial_guess=x0, tolerance=tolerance, max_iterations=max_iter)
            elapsed = time.perf_counter() - start_time
            
            # Compute A-norm errors
            error_A = [np.sqrt((xk-x_star).T @ A @ (xk-x_star)) for xk in history]
            
            final_error = error_A[-1]
            time_per_iter = (elapsed / iters) * 1000  # milliseconds per iteration
            
            # Print metrics
            print(f"{n:<6} {kappa:<6} {iters:<12} {elapsed:<10.4f} {final_error:<15.2e} {time_per_iter:<15.2f}")

            # Use loglog instead of semilogy
            ax.loglog(
                np.arange(1, len(error_A) + 1),  # Start from 1 to avoid log(0)
                error_A,
                label=f'κ={kappa}', 
                marker='.', 
                markevery=5
            )
        
        # Finalize plot
        ax.grid(True, which='both', linestyle='--', alpha=0.7)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('A-Norm Error')
        ax.set_title(f'CG A-Norm Error Convergence (n={n})')
        ax.legend()
        
        # Save plot
        plt.savefig(os.path.join(results_dir, f'cg_anorm_n{n}.png'), dpi=300)
        plt.close()
        print(f"Saved plot: cg_anorm_n{n}.png")

if __name__ == "__main__":
    np.random.seed(42)
    run_cg_error_experiments(
        sizes=(10, 100, 200, 1000),
        cond_nums=(10, 100,200, 1000),
        tolerance=1e-8,
        max_iter=10000
    )
