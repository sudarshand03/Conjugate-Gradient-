# experiments/sd_convergence.py

import os
import sys

# 1) allow imports from project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, project_root)

import numpy as np
import matplotlib.pyplot as plt

from resources.generate_spd import make_spd_matrix
from models.steepest_descent import steepest_descent
from resources.plot_utils import apply_default_style

def estimate_rate(residuals, burn_in=5):
    """
    Estimate asymptotic convergence rate r ≈ mean(‖r_{k+1}‖/‖r_k‖)
    over the tail (after burn_in iterations).
    """
    ratios = [
        residuals[k+1] / residuals[k]
        for k in range(burn_in, len(residuals)-1)
        if residuals[k] > 0
    ]
    return float(np.mean(ratios)) if ratios else np.nan

def run_sd_experiments(
    sizes=(10, 100, 1000),
    cond_nums=(10, 100, 1000, 10000),
    tol=1e-8
):
    # make results dir
    results_dir = os.path.abspath("results")
    os.makedirs(results_dir, exist_ok=True)

    apply_default_style()

    for n in sizes:
        fig, ax = plt.subplots(figsize=(8, 6))

        for kappa in cond_nums:
            # 2) build SPD test problem
            A = make_spd_matrix(
                n_dim=n,
                condition_number=float(kappa),
                distribution="linear",
                random_state=42
            )
            rng = np.random.default_rng(42)
            b = rng.standard_normal(n)
            b /= np.linalg.norm(b)
            x0 = np.zeros(n)

            # 3) run Steepest Descent
            x_sd, history, its, _ = steepest_descent(
                A, b, x0,
                tolerance=tol,
                max_iterations=n,
                store_history=True
            )

            # 4) compute energy-norm residuals: ‖r‖_A = sqrt(rᵀ A r)
            energy_res = []
            for xk in history:
                r = b - A.dot(xk)
                energy_res.append(np.sqrt(r.dot(A.dot(r))))

            # 5) estimate rate
            r_est = estimate_rate(energy_res)
            print(f"n={n}, κ={kappa}: its={its}, rate≈{r_est:.3f}")

            # 6) plot
            ax.semilogy(
                range(len(energy_res)),
                energy_res,
                marker='o',
                linestyle='-',
                label=f'κ={kappa}, r≈{r_est:.3f}'
            )

        ax.set_title(f'SD Convergence in Energy Norm (n={n})')
        ax.set_xlabel('Iteration k')
        ax.set_ylabel(r'$\|r_k\|_A = \sqrt{r_k^\top A\,r_k}$')
        ax.grid(which='both', linestyle='--')
        ax.legend(loc='best')
        fig.tight_layout()

        # 7) save
        out_file = os.path.join(results_dir, f'sd_convergence_n{n}.png')
        fig.savefig(out_file, dpi=300)
        plt.close(fig)
        print(f"Saved plot for n={n} → {out_file}")

if __name__ == "__main__":
    run_sd_experiments()
