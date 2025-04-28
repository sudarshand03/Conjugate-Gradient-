import numpy as np
import matplotlib.pyplot as plt
from Core_Functions.conjugate_gradient import conjugate_gradient
from Core_Functions.mean_variance import build_mvot_system, solve_mvot_sd, solve_mvot_cg
from Core_Functions.ActiveManagement import fetch_stock_data, momentum_calculator, covariance_matrix_build
from scipy.linalg import *
from Core_Functions.spd_generate import generate_spd_matrix
from Core_Functions.steepest_descent import steepest_descent, plot_residual_diagram
import time
import os

# Configure Matplotlib for clear, log-scaled plots
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams.update({
    'font.size': 12, 'lines.linewidth': 2, 'axes.labelsize': 14,
    'axes.titlesize': 16, 'legend.fontsize': 12, 'figure.dpi': 300
})

# Set random seed for reproducibility
np.random.seed(42)

def run_experiment(n: int, condition_number: float, tolerance: float = 1e-8, max_iter: int = 50000):
    """
    Run Steepest Descent and Conjugate Gradient experiments.

    Args:
        n (int): Matrix size.
        condition_number (float): Condition number for matrix generation.
        tolerance (float): Convergence tolerance.
        max_iter (int): Maximum iterations.

    Returns:
        Tuple of iteration counts, times, final residuals, and residual histories.
    """
    A = generate_spd_matrix(n, condition_number)
    b = np.random.randn(n)
    b = b / np.linalg.norm(b)  # Normalize b to unit norm
    x0 = np.zeros(n)

    # Steepest Descent
    start_time = time.time()
    x_sd, iterates_sd, iters_sd, _ = steepest_descent(A, b, x0, tolerance, max_iter)
    sd_time = time.time() - start_time
    residuals_sd = [np.linalg.norm(b - A @ x_k) for x_k in iterates_sd]

    # Conjugate Gradient
    start_time = time.time()
    x_cg, iterates_cg, iters_cg, _ = conjugate_gradient(A, b, x0, tolerance, max_iter)
    cg_time = time.time() - start_time
    residuals_cg = [np.linalg.norm(b - A @ x_k) for x_k in iterates_cg]

    return iters_sd, sd_time, residuals_sd[-1], iters_cg, cg_time, residuals_cg[-1], residuals_sd, residuals_cg

def test_mvot(n: int, condition_number: float, mu_p: float = 0.1, tolerance: float = 1e-8, max_iter: int = 10000):
    """
    Build and execute synthetic MVOT tests using SD and CG, returning results (maybe for plotting)

    Args:
        n (int): matrix size (number of assets)
        mu_p (float): target portfolio returns
        condition_number (float): condition number for SPD matrix
        tolerance (float): convergence tolerance (default: 1e-8)
        max_iter (int): maximum iterations (increased to 10000)

    Returns:
        Tuple: 
    """
    # generate synthetic covariance matrix
    Sigma = generate_spd_matrix(n, condition_number)

    # generate synthetic expected returns
    mu = np.random.uniform(0.05, 0.15, size=n)

    # Build spd reduced system
    A_reduced, b_reduced, x0, y0, Z = build_mvot_system(Sigma,mu,mu_p,n)

    # Solve using Steepest Descent
    start_sd = time.perf_counter()
    y_sd, _, iters_sd, _ = steepest_descent(A_reduced, b_reduced, y0, tolerance=1e-10, max_iterations=10000)
    x_sd = x0 + Z @ y_sd
    end_sd = time.perf_counter()
    time_sd = end_sd - start_sd

    # Solve using Conjugate Gradient
    start_cg = time.perf_counter()
    y_cg, _, iters_cg, _ = conjugate_gradient(A_reduced, b_reduced, y0, tolerance=1e-10, max_iterations=10000)
    x_cg = x0 + Z @ y_cg
    end_cg = time.perf_counter()
    time_cg = end_cg - start_cg

    return {"Portfolio Weights (sd):": x_sd.flatten(),
            "Portfolio Weights (cg):": x_cg.flatten(),
            "Iterations (sd):": iters_sd,
            "Iterations (cg):": iters_cg,
            "Expected Return (sd):": mu @ x_sd,
            "Expected Return (cg):": mu @ x_cg,
            "Expected Return Error (sd):": abs(mu @ x_sd - mu_p),
            "Expected Return Error (cg):": abs(mu @ x_cg - mu_p),
            "Sum of Weights (sd):": np.sum(x_sd),
            "Sum of Weights (cg):": np.sum(x_cg),
            "Weight Sum Error (sd):": abs(np.sum(x_sd) - 1.0),
            "Weight Sum Error (cg):": abs(np.sum(x_cg) - 1.0),
            "Time to converge (sd):": time_sd,
            "Time to converge (cg):": time_cg}

def test_mvot_real(tickers: list, n: int, condition_number: float, mu_p: float = 0.1, tolerance: float = 1e-8, max_iter: int = 10000):
    """
    Build and execute real MVOT tests using SD and CG, returning results (maybe for plotting)

    Args:
        tickers (list): list of stock tickers
        n (int): matrix size (number of assets)
        mu_p (float): target portfolio returns
        condition_number (float): condition number for SPD matrix
        tolerance (float): convergence tolerance (default: 1e-8)
        max_iter (int): maximum iterations (increased to 10000)

    Returns:
        Tuple: 
    """
    # fetch data
    price_data = fetch_stock_data(tickers)
    price_data = price_data.dropna(axis=1)

    # set calibration data for covariance matrix
    calibration_data = price_data.iloc[-252:]

    # get momentum data for expected return
    mu_dict = momentum_calculator(calibration_data)
    mu = np.array(list(mu_dict.values()))

    # calculate covariance matrix
    Sigma = covariance_matrix_build(calibration_data).to_numpy()

    # set target return
    if mu_p is None:
        mu_p = mu.mean()

    # Build spd reduced system
    A_reduced, b_reduced, x0, y0, Z = build_mvot_system(Sigma,mu,mu_p,n)

    # Solve using Steepest Descent
    start_sd = time.perf_counter()
    y_sd, _, iters_sd, _ = steepest_descent(A_reduced, b_reduced, y0, tolerance=1e-10, max_iterations=10000)
    x_sd = x0 + Z @ y_sd
    end_sd = time.perf_counter()
    time_sd = end_sd - start_sd

    # Solve using Conjugate Gradient
    start_cg = time.perf_counter()
    y_cg, _, iters_cg, _ = conjugate_gradient(A_reduced, b_reduced, y0, tolerance=1e-10, max_iterations=10000)
    x_cg = x0 + Z @ y_cg
    end_cg = time.perf_counter()
    time_cg = end_cg - start_cg

    return {"Portfolio Weights (sd):": x_sd.flatten(),
            "Portfolio Weights (cg):": x_cg.flatten(),
            "Iterations (sd):": iters_sd,
            "Iterations (cg):": iters_cg,
            "Expected Return (sd):": mu @ x_sd,
            "Expected Return (cg):": mu @ x_cg,
            "Expected Return Error (sd):": abs(mu @ x_sd - mu_p),
            "Expected Return Error (cg):": abs(mu @ x_cg - mu_p),
            "Sum of Weights (sd):": np.sum(x_sd),
            "Sum of Weights (cg):": np.sum(x_cg),
            "Weight Sum Error (sd):": abs(np.sum(x_sd) - 1.0),
            "Weight Sum Error (cg):": abs(np.sum(x_cg) - 1.0),
            "Time to converge (sd):": time_sd,
            "Time to converge (cg):": time_cg}

def eigenvalue_test(n: int, eigenvalues: list, tolerance: float = 1e-8, max_iter: int = 10000):
    """
    Test CG with specific eigenvalues and plot convergence on log-log scale.

    Args:
        n (int): Matrix size.
        eigenvalues (list): List of eigenvalues to use.
        perturb (bool): Whether to perturb eigenvalues.
        tolerance (float): Convergence tolerance.
        max_iter (int): Maximum iterations.

    Returns:
        Tuple of iterations to convergence and final residual.
    """
    A = generate_spd_matrix(n, eigenvalues=eigenvalues, perturb=perturb)
    b = np.random.randn(n)
    b = b / np.linalg.norm(b)  # Normalize b
    x0 = np.zeros(n)
    x_cg, iterates_cg, iters_cg, _ = conjugate_gradient(A, b, x0, tolerance, max_iter)
    residuals_cg = [np.linalg.norm(b - A @ x_k) for x_k in iterates_cg]
    actual_iters = next(i for i, res in enumerate(residuals_cg) if res < tolerance)

    # Plot convergence on log-log scale
    plt.figure(figsize=(10, 6))
    plt.loglog(range(1, len(residuals_cg) + 1), residuals_cg, label='Conjugate Gradient', 
               color='tab:green', marker='^', markevery=max(1, actual_iters//5))
    plt.xlabel('Iteration (Log Scale)')
    plt.ylabel('Residual Norm ($||r_k||$) (Log Scale)')
    plt.title(f'CG with {"Perturbed " if perturb else ""}Eigenvalues: {eigenvalues}')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.savefig(os.path.join("results", f"cg_eig_n{n}{'_perturbed' if perturb else ''}.png"), dpi=300)
    plt.close()

    return actual_iters, residuals_cg[actual_iters]

if __name__ == "__main__":
    # Ensure results directory exists
    os.makedirs("results", exist_ok=True)

    # Small-scale test (n=2) with log-log plot and residual diagram
    A = np.array([[4, 1], [1, 3]], dtype=float)
    b = np.array([1, 2], dtype=float)
    x0 = np.zeros(2)
    x_sd, iterates_sd, iters_sd, _ = steepest_descent(A, b, x0)
    x_cg, iterates_cg, iters_cg, _ = conjugate_gradient(A, b, x0)
    residuals_sd = [np.linalg.norm(b - A @ x_k) for x_k in iterates_sd]
    residuals_cg = [np.linalg.norm(b - A @ x_k) for x_k in iterates_cg]
    plt.figure(figsize=(10, 6))
    plt.loglog(range(1, len(residuals_sd) + 1), residuals_sd, label='Steepest Descent', 
               color='tab:blue', marker='o')
    plt.loglog(range(1, len(residuals_cg) + 1), residuals_cg, label='Conjugate Gradient', 
               color='tab:orange', marker='s')
    plt.xlabel('Iteration (Log Scale)')
    plt.ylabel('Residual Norm ($||r_k||$) (Log Scale)')
    plt.title('SD vs. CG: $n = 2$')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.savefig(os.path.join("results", "small_scale.png"), dpi=300)
    plt.close()
    plot_residual_diagram(A, b, iterates_sd, os.path.join("results", "residual_diagram_n2.png"))

    # Main experiments across different sizes and condition numbers
    sizes = [10, 100, 1000]
    condition_numbers = [10, 100, 1000, 10000]
    results = []
    all_residuals_sd = {n: {} for n in sizes}
    all_residuals_cg = {n: {} for n in sizes}
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

    for n in sizes:
        for i, kappa in enumerate(condition_numbers):
            iters_sd, sd_time, res_sd, iters_cg, cg_time, res_cg, residuals_sd, residuals_cg = \
                run_experiment(n, kappa)
            results.append((n, kappa, iters_sd, sd_time, res_sd, iters_cg, cg_time, res_cg))
            all_residuals_sd[n][kappa] = residuals_sd
            all_residuals_cg[n][kappa] = residuals_cg

    # Combined convergence plots with log-log scales
    for n in sizes:
        plt.figure(figsize=(12, 8))
        for i, kappa in enumerate(condition_numbers):
            plt.loglog(range(1, len(all_residuals_sd[n][kappa]) + 1), all_residuals_sd[n][kappa], 
                       '--', color=colors[i], label=f'SD (κ={kappa})', alpha=0.7)
            plt.loglog(range(1, len(all_residuals_cg[n][kappa]) + 1), all_residuals_cg[n][kappa], 
                       '-', color=colors[i], label=f'CG (κ={kappa})')
        plt.xlabel('Iteration (Log Scale)')
        plt.ylabel('Residual Norm ($||r_k||$) (Log Scale)')
        plt.title(f'SD vs CG Convergence (n={n})')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, which="both", ls="--")
        plt.tight_layout()
        plt.savefig(os.path.join("results", f"convergence_n{n}.png"), dpi=300, bbox_inches='tight')
        plt.close()

    # Performance profile with log-log scale
    plt.figure(figsize=(10, 6))
    for i, kappa in enumerate(condition_numbers):
        cg_times = [r[6] for r in results if r[1] == kappa]
        sd_times = [r[3] for r in results if r[1] == kappa]
        sizes_subset = [r[0] for r in results if r[1] == kappa]
        plt.loglog(sizes_subset, cg_times, '-o', color=colors[i], label=f'CG (κ={kappa})')
        plt.loglog(sizes_subset, sd_times, '--o', color=colors[i], label=f'SD (κ={kappa})', alpha=0.7)
    plt.xlabel('Matrix Size (n) (Log Scale)')
    plt.ylabel('Time (seconds) (Log Scale)')
    plt.title('Performance Profile')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.savefig(os.path.join("results", "performance_profile.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # Eigenvalue test with 5 distinct eigenvalues
    iters_eig, res_eig = eigenvalue_test(100, [1, 2, 3, 4, 5], perturb=False)
    print(f"Eigenvalue Test (n=100, exact): CG {iters_eig} iters (residual {res_eig:.2e})")

    # Eigenvalue test with perturbed eigenvalues
    iters_eig_pert, res_eig_pert = eigenvalue_test(100, [1, 2, 3, 4, 5], perturb=True)
    print(f"Eigenvalue Test (n=100, perturbed): CG {iters_eig_pert} iters (residual {res_eig_pert:.2e})")

    # Non-symmetric matrix test
    A_non_sym = generate_spd_matrix(100, 10.0, epsilon=0.1)
    b = np.random.randn(100)
    b = b / np.linalg.norm(b)
    x0 = np.zeros(100)
    try:
        x_cg, _, iters_cg, _ = conjugate_gradient(A_non_sym, b, x0)
        print(f"CG with non-symmetric A converged in {iters_cg} iterations (unexpected)")
    except ValueError as e:
        print(f"CG failed for non-symmetric A as expected: {e}")

    # Print experiment results
    print("\nExperiment Results:")
    for r in results:
        print(f"n={r[0]}, κ={r[1]}: SD {r[2]} iters ({r[3]:.4f}s, {r[4]:.2e}), CG {r[5]} iters ({r[6]:.4f}s, {r[7]:.2e})")
    print(f"Eigenvalue Test (n=100): CG {iters_eig} iters ({res_eig:.2e})")

    # Synthetic MVOT system tests
    print("\n ----- Synthetic MVOT System Tests -----")
    n_mvot = [5, 10, 20, 50]
    for n in n_mvot:
        result = test_mvot(n, condition_number=100)
        print(f"\nSynthetic MVOT Result (n={n}, κ=100):")
        print(f"  SD -> iters: {result['Iterations (sd):']} | time: {result['Time to converge (sd):']} | μᵀx: {result['Expected Return (sd):']:.4f} | err: {result['Expected Return Error (sd):']:.2e} | Σx={result['Sum of Weights (sd):']:.4f} | err: {result['Weight Sum Error (sd):']:.2e}")
        print(f"  CG -> iters: {result['Iterations (cg):']} | time: {result['Time to converge (cg):']} | μᵀx: {result['Expected Return (cg):']:.4f} | err: {result['Expected Return Error (cg):']:.2e} | Σx={result['Sum of Weights (cg):']:.4f} | err: {result['Weight Sum Error (cg):']:.2e}")
        print(f"  Weights (SD): {np.round(result['Portfolio Weights (sd):'], 4)}")
        print(f"  Weights (CG): {np.round(result['Portfolio Weights (cg):'], 4)}")

    # Real MVOT system tests
    print("\n ----- Real MVOT System Tests -----")
    n_mvot_real = [5, 10, 20, 50]
    tickers = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "UNH", "JPM",
        "JNJ", "V", "PG", "MA", "HD", "XOM", "LLY", "ABBV", "CVX", "MRK",
        "PEP", "KO", "BAC", "AVGO", "WMT", "PFE", "COST", "TMO", "CSCO", "MCD",
        "DIS", "ACN", "ADBE", "CRM", "ABT", "DHR", "INTC", "NKE", "TXN", "NEE",
        "LIN", "AMD", "AMGN", "QCOM", "LOW", "UPS", "MS", "UNP", "PM", "RTX"]
    for n in n_mvot_real:
        result = test_mvot_real(tickers[:n], n, condition_number=100)
        print(f"\nReal MVOT Result (n={n}, κ=100):")
        print(f"  SD -> iters: {result['Iterations (sd):']} | time: {result['Time to converge (sd):']} | μᵀx: {result['Expected Return (sd):']:.4f} | err: {result['Expected Return Error (sd):']:.2e} | Σx={result['Sum of Weights (sd):']:.4f} | err: {result['Weight Sum Error (sd):']:.2e}")
        print(f"  CG -> iters: {result['Iterations (cg):']} | time: {result['Time to converge (cg):']} | μᵀx: {result['Expected Return (cg):']:.4f} | err: {result['Expected Return Error (cg):']:.2e} | Σx={result['Sum of Weights (cg):']:.4f} | err: {result['Weight Sum Error (cg):']:.2e}")
        print(f"  Weights (SD): {np.round(result['Portfolio Weights (sd):'], 4)}")
        print(f"  Weights (CG): {np.round(result['Portfolio Weights (cg):'], 4)}")

        print(f"n={r[0]}, κ={r[1]}: SD {r[2]} iters ({r[3]:.4f}s, residual {r[4]:.2e}), "
              f"CG {r[5]} iters ({r[6]:.4f}s, residual {r[7]:.2e})")