import numpy as np
import matplotlib.pyplot as plt
from Core_Functions.spd_generate import generate_spd_matrix
from Core_Functions.steepest_descent import steepest_descent
from Core_Functions.conjugate_gradient import conjugate_gradient
from Core_Functions.mean_variance import build_mvot_system, solve_mvot_sd, solve_mvot_cg
from Core_Functions.ActiveManagement import fetch_stock_data, momentum_calculator, covariance_matrix_build
from scipy.linalg import *
import time

# Configure Matplotlib for high-quality plots
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams.update({
    'font.size': 12, 'lines.linewidth': 2, 'axes.labelsize': 14,
    'axes.titlesize': 16, 'legend.fontsize': 12, 'figure.dpi': 300
})

def run_experiment(n: int, condition_number: float, tolerance: float = 1e-8, max_iter: int = 10000):
    """
    Execute SD and CG experiments, returning results for plotting.

    Args:
        n (int): Matrix size.
        condition_number (float): Condition number for SPD matrix.
        tolerance (float): Convergence tolerance (default: 1e-8).
        max_iter (int): Maximum iterations (increased to 10000).

    Returns:
        Tuple: SD and CG iterations, times, final residuals, residual histories.
    """
    # Setup problem
    A: np.ndarray = generate_spd_matrix(n, condition_number)
    b: np.ndarray = np.random.randn(n)
    x0: np.ndarray = np.zeros(n)

    # Run Steepest Descent
    start_time: float = time.time()
    x_sd, iterates_sd, iters_sd, x_star = steepest_descent(A, b, x0, tolerance, max_iter)
    sd_time: float = time.time() - start_time
    residuals_sd: list = [np.linalg.norm(b - A @ x_k) for x_k in iterates_sd]

    # Run Conjugate Gradient
    start_time = time.time()
    x_cg, iterates_cg, iters_cg, _ = conjugate_gradient(A, b, x0, tolerance, max_iter)
    cg_time: float = time.time() - start_time
    residuals_cg: list = [np.linalg.norm(b - A @ x_k) for x_k in iterates_cg]

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
    Test CG with specific eigenvalues, plotting residuals.

    Args:
        n (int): Matrix size.
        eigenvalues (list): List of eigenvalues.
        tolerance (float): Convergence tolerance.
        max_iter (int): Maximum iterations.

    Returns:
        Tuple: CG iterations, final residual.
    """
    A: np.ndarray = generate_spd_matrix(n, eigenvalues=eigenvalues)
    b: np.ndarray = np.random.randn(n)
    x0: np.ndarray = np.zeros(n)
    x_cg, iterates_cg, iters_cg, x_star = conjugate_gradient(A, b, x0, tolerance, max_iter)
    residuals_cg: list = [np.linalg.norm(b - A @ x_k) for x_k in iterates_cg]

    plt.figure(figsize=(10, 6))
    plt.semilogy(residuals_cg, label='Conjugate Gradient', color='tab:green', marker='^', markevery=max(1, iters_cg//5))
    plt.xlabel('Iteration')
    plt.ylabel('Residual Norm ($||r_k||$)')
    plt.title(f'CG with Eigenvalues: {eigenvalues}')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.savefig(f"cg_eig_n{n}.png", dpi=300)
    plt.close()

    return iters_cg, residuals_cg[-1]

if __name__ == "__main__":
    # Small-scale test
    A: np.ndarray = np.array([[4, 1], [1, 3]], dtype=float)
    b: np.ndarray = np.array([1, 2], dtype=float)
    x0: np.ndarray = np.zeros(2)
    x_sd, iterates_sd, iters_sd, _ = steepest_descent(A, b, x0)
    x_cg, iterates_cg, iters_cg, _ = conjugate_gradient(A, b, x0)
    residuals_sd: list = [np.linalg.norm(b - A @ x_k) for x_k in iterates_sd]
    residuals_cg: list = [np.linalg.norm(b - A @ x_k) for x_k in iterates_cg]
    plt.figure(figsize=(10, 6))
    plt.semilogy(residuals_sd, label='Steepest Descent', color='tab:blue', marker='o')
    plt.semilogy(residuals_cg, label='Conjugate Gradient', color='tab:orange', marker='s')
    plt.xlabel('Iteration')
    plt.ylabel('Residual Norm ($||r_k||$)')
    plt.title('SD vs. CG: $n = 2$')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.savefig("small_scale.png", dpi=300)
    plt.close()

    # Main experiments
    sizes: list = [10, 100, 1000]
    condition_numbers: list = [10, 100, 1000, 10000]
    results: list = []
    all_residuals_sd: dict = {n: {} for n in sizes}
    all_residuals_cg: dict = {n: {} for n in sizes}

    # Colors for different condition numbers
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

    # Run experiments and store residuals
    for n in sizes:
        for i, kappa in enumerate(condition_numbers):
            iters_sd, sd_time, res_sd, iters_cg, cg_time, res_cg, residuals_sd, residuals_cg = run_experiment(n, kappa)
            results.append((n, kappa, iters_sd, sd_time, res_sd, iters_cg, cg_time, res_cg))
            all_residuals_sd[n][kappa] = residuals_sd
            all_residuals_cg[n][kappa] = residuals_cg

    # Create one combined plot per matrix size
    for n in sizes:
        plt.figure(figsize=(12, 8))
        for i, kappa in enumerate(condition_numbers):
            plt.semilogy(all_residuals_sd[n][kappa], '--', color=colors[i], label=f'SD (κ={kappa})', alpha=0.7)
            plt.semilogy(all_residuals_cg[n][kappa], '-', color=colors[i], label=f'CG (κ={kappa})')
        plt.xlabel('Iteration')
        plt.ylabel('Residual Norm ($||r_k||$)')
        plt.title(f'SD vs CG Convergence (n={n})')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, which="both", ls="--")
        plt.tight_layout()
        plt.savefig(f"convergence_n{n}.png", dpi=300, bbox_inches='tight')
        plt.close()

    # Performance profile plot
    plt.figure(figsize=(10, 6))
    for i, kappa in enumerate(condition_numbers):
        cg_times = [r[6] for r in results if r[1] == kappa]
        sd_times = [r[3] for r in results if r[1] == kappa]
        sizes_subset = [r[0] for r in results if r[1] == kappa]
        plt.loglog(sizes_subset, cg_times, '-o', color=colors[i], label=f'CG (κ={kappa})')
        plt.loglog(sizes_subset, sd_times, '--o', color=colors[i], label=f'SD (κ={kappa})', alpha=0.7)
    plt.xlabel('Matrix Size (n)')
    plt.ylabel('Time (seconds)')
    plt.title('Performance Profile')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.savefig("performance_profile.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Eigenvalue test
    iters_eig, res_eig = eigenvalue_test(100, [1, 2, 3, 4, 5])

    # Output numerical results
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
