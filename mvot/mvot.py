#!/usr/bin/env python3
"""
mvot_runner.py
--------------
Run synthetic-data and real-market mean–variance-optimal-portfolio (MVOT) experiments
with Steepest-Descent (SD) and Conjugate-Gradient (CG) solvers.

Requirements
------------
• models.conjugate_gradient.conjugate_gradient
• models.steepest_descent.steepest_descent
• mvot.mean_variance.build_mvot_system
• mvot.ActiveManagement.{fetch_stock_data,momentum_calculator,covariance_matrix_build}
• resources.generate_spd.generate_spd_matrix
"""

import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

import time
from typing import Dict, List

import numpy as np

from models.conjugate_gradient import conjugate_gradient
from models.steepest_descent import steepest_descent
from mvot.mean_variance import build_mvot_system
from mvot.ActiveManagement import (
    fetch_stock_data,
    momentum_calculator,
    covariance_matrix_build,
)
from mvot.spd_generate import generate_spd_matrix


# -----------------------------------------------------------------------------#
#                               MVOT TEST HELPERS                               #
# -----------------------------------------------------------------------------#
def test_mvot_synthetic(
    n: int,
    condition_number: float,
    mu_p: float = 0.1,
    tol: float = 1e-8,
    max_iter: int = 10_000,
) -> Dict[str, float | np.ndarray | int]:
    """Solve a synthetic MVOT instance with SD & CG and return diagnostic stats."""
    # Synthetic Σ and μ
    Sigma = generate_spd_matrix(n, condition_number)
    mu = np.random.uniform(0.05, 0.15, size=n)

    # Build reduced (SPD) system   ZᵀΣZ  y = Zᵀ(λ·1 – Σx₀)
    A_red, b_red, x0, y0, Z = build_mvot_system(Sigma, mu, mu_p, n)

    # --- Steepest Descent -----------------------------------------------------
    t0 = time.perf_counter()
    y_sd, _, iters_sd, _ = steepest_descent(A_red, b_red, y0, tol, max_iter)
    sd_time = time.perf_counter() - t0
    x_sd = x0 + Z @ y_sd

    # --- Conjugate Gradient ---------------------------------------------------
    t0 = time.perf_counter()
    y_cg, _, iters_cg, _ = conjugate_gradient(A_red, b_red, y0, tol, max_iter)
    cg_time = time.perf_counter() - t0
    x_cg = x0 + Z @ y_cg

    return _collect_stats(mu, mu_p, x_sd, x_cg, iters_sd, iters_cg, sd_time, cg_time)


def test_mvot_real(
    tickers: List[str],
    mu_p: float | None = None,
    tol: float = 1e-8,
    max_iter: int = 10_000,
) -> Dict[str, float | np.ndarray | int]:
    """Solve an MVOT instance built from historical price data."""
    prices = fetch_stock_data(tickers).dropna(axis=1)
    calib = prices.iloc[-252:]  # one trading year

    # μ from 12-month momentum; Σ from sample covariance
    mu = np.array(list(momentum_calculator(calib).values()))
    Sigma = covariance_matrix_build(calib).to_numpy()

    if mu_p is None:
        mu_p = float(mu.mean())

    # Build reduced system
    n = len(tickers)
    A_red, b_red, x0, y0, Z = build_mvot_system(Sigma, mu, mu_p, n)

    # --- SD -------------------------------------------------------------------
    t0 = time.perf_counter()
    y_sd, _, iters_sd, _ = steepest_descent(A_red, b_red, y0, tol, max_iter)
    sd_time = time.perf_counter() - t0
    x_sd = x0 + Z @ y_sd

    # --- CG -------------------------------------------------------------------
    t0 = time.perf_counter()
    y_cg, _, iters_cg, _ = conjugate_gradient(A_red, b_red, y0, tol, max_iter)
    cg_time = time.perf_counter() - t0
    x_cg = x0 + Z @ y_cg

    return _collect_stats(mu, mu_p, x_sd, x_cg, iters_sd, iters_cg, sd_time, cg_time)


# -----------------------------------------------------------------------------#
#                                UTILITIES                                      #
# -----------------------------------------------------------------------------#
def _collect_stats(
    mu: np.ndarray,
    mu_p: float,
    x_sd: np.ndarray,
    x_cg: np.ndarray,
    iters_sd: int,
    iters_cg: int,
    sd_time: float,
    cg_time: float,
) -> Dict[str, float | np.ndarray | int]:
    """Package common diagnostics into a single dict."""
    return {
        "Portfolio Weights (SD)": x_sd.flatten(),
        "Portfolio Weights (CG)": x_cg.flatten(),
        "Iterations (SD)": iters_sd,
        "Iterations (CG)": iters_cg,
        "μᵀx (SD)": float(mu @ x_sd),
        "μᵀx (CG)": float(mu @ x_cg),
        "Target Return μ_p": mu_p,
        "Return Error (SD)": abs(mu @ x_sd - mu_p),
        "Return Error (CG)": abs(mu @ x_cg - mu_p),
        "Σ Weights (SD)": float(x_sd.sum()),
        "Σ Weights (CG)": float(x_cg.sum()),
        "Weight Sum Error (SD)": abs(x_sd.sum() - 1.0),
        "Weight Sum Error (CG)": abs(x_cg.sum() - 1.0),
        "Time (SD)": sd_time,
        "Time (CG)": cg_time,
    }


# -----------------------------------------------------------------------------#
#                                   MAIN                                        #
# -----------------------------------------------------------------------------#
if __name__ == "__main__":
    np.random.seed(42)
    os.makedirs("results", exist_ok=True)

    # -------------------------- Synthetic Tests ------------------------------ #
    print("\n─── Synthetic MVOT Experiments ───")
    for n in (5, 10, 20, 50):
        stats = test_mvot_synthetic(n, condition_number=100)
        print(f"\nn = {n}, κ = 100")
        print(f"  SD → iters {stats['Iterations (SD)']:>5} | "
              f"time {stats['Time (SD)']:.4f}s | μᵀx {stats['μᵀx (SD)']:.4f} "
              f"| Σw {stats['Σ Weights (SD)']:.4f}")
        print(f"  CG → iters {stats['Iterations (CG)']:>5} | "
              f"time {stats['Time (CG)']:.4f}s | μᵀx {stats['μᵀx (CG)']:.4f} "
              f"| Σw {stats['Σ Weights (CG)']:.4f}")

    # --------------------------- Real-Market Tests --------------------------- #
    tickers = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "UNH", "JPM",
        "JNJ",  "V",    "PG",    "MA",   "HD",   "XOM",  "LLY",  "ABBV",  "CVX", "MRK",
        "PEP",  "KO",   "BAC",   "AVGO", "WMT",  "PFE",  "COST", "TMO",   "CSCO", "MCD",
        "DIS",  "ACN",  "ADBE",  "CRM",  "ABT",  "DHR",  "INTC", "NKE",   "TXN",  "NEE",
        "LIN",  "AMD",  "AMGN",  "QCOM", "LOW",  "UPS",  "MS",   "UNP",   "PM",   "RTX",
    ]

    print("\n─── Real-Market MVOT Experiments ───")
    for n in (5, 10, 20, 50):
        stats = test_mvot_real(tickers[:n])
        print(f"\nn = {n} tickers")
        print(f"  SD → iters {stats['Iterations (SD)']:>5} | "
              f"time {stats['Time (SD)']:.4f}s | μᵀx {stats['μᵀx (SD)']:.4f} "
              f"| Σw {stats['Σ Weights (SD)']:.4f}")
        print(f"  CG → iters {stats['Iterations (CG)']:>5} | "
              f"time {stats['Time (CG)']:.4f}s | μᵀx {stats['μᵀx (CG)']:.4f} "
              f"| Σw {stats['Σ Weights (CG)']:.4f}")
