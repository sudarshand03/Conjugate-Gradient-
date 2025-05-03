# experiments/cg_error_convergence.py

import os
import sys

# 1) Make project root importable
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

import numpy as np
import matplotlib.pyplot as plt

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
    results_dir = os.path.abspath("results")
    os.makedirs(results_dir, exist_ok=True)

    # apply_default_style()
    print("\n Conjugate Gradient A-Norm Error Metrics")
    print("--"*70)
    print(f"{n:<6} {'kappa':<6} {'iters':<12} {'elapsed':<10.4f} {'final_error':<15.2e} {'time_per_iter':<15.2f}")


    for n in sizes:
        fig, ax = plt.subplots(figsize=(8, 6))

        for kappa in cond_nums:
            # build SPD test matrix with condition number κ
            A = generate_spd(
                n_dim=n,
                condition_number=float(kappa),
                distribution="log",
                random_state=42
            )
  
            # create a consistent RHS
            rng = np.random.default_rng(42)
            b = rng.standard_normal(n)
            b /= np.linalg.norm(b)
            x0 = np.zeros(n)

            # run CG, capturing all iterates
            x_cg, iterates, its, x_star = conjugate_gradient(
                matrix=A,
                rhs=b,
                initial_guess=x0,
                tolerance=tolerance,
                max_iterations=max_iter
            )

            # compute A-norm of the error: ||e_k||_A = sqrt((x_k - x*)ᵀ A (x_k - x*))
            error_A = [
                np.sqrt((xk - x_star).dot(A.dot(xk - x_star)))
                for xk in iterates
            ]
            
            
            # print(f"{n:<6} {kappa:<6} {iters:<12} {elapsed:<10.4f} {final_error:<15.2e} {time_per_iter:<15.2f}")

            # choose marker spacing to avoid clutter
            markevery = max(1, len(error_A) // 20)

            # plot on a semilogy scale
            ax.semilogy(
                np.arange(len(error_A)),
                error_A,
                marker='s',
                markevery=markevery,
                linewidth=2,
                label=f"κ={kappa}"
            )

        ax.set_title(f"CG Error Convergence in A-norm (n={n})", pad=14)
        ax.set_xlabel("Iteration k")
        ax.set_ylabel(r"Error $\|x_k - x^*\|_A$")
        ax.grid(True, which="both", linestyle="--", alpha=0.7)
        ax.legend(title="Condition Number", loc="upper right")
        fig.tight_layout()

        out_path = os.path.join(results_dir, f"cg_error_convergence_n{n}.png")
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"Saved CG A-norm error plot for n={n} → {out_path}")


if __name__ == "__main__":
    run_cg_error_experiments(
        sizes=(10, 100, 1000),
        cond_nums=(10, 100, 1000),
        tolerance=1e-8,
        max_iter=10000
    )
