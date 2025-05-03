# #!/usr/bin/env python3
# from __future__ import annotations
# from datetime import date
# from typing import Sequence, Tuple

# import numpy as np
# import pandas as pd
# import yfinance as yf
# import seaborn as sns
# import matplotlib.pyplot as plt

# from projected_cg import projected_conjugate_gradient


# def fetch_stock_data(
#     tickers: Sequence[str],
#     start: str = "2024-12-20",
#     end: str = date.today().isoformat(),
#     auto_adjust: bool = False,
# ) -> pd.DataFrame:

#     df = yf.download(
#         tickers, start=start, end=end, progress=False, auto_adjust=auto_adjust
#     )["Close"]
#     df = df.dropna(axis=1)
#     return df.pct_change().dropna()


# def solve_markowitz(
#     returns: pd.DataFrame, target_return: float, tol: float = 1e-8
# ) -> Tuple[np.ndarray, np.ndarray]:

#     mu = returns.mean().values
#     sigma = returns.cov().values
#     n = mu.size

#     # Build constraints A x = c
#     A = np.vstack([mu, np.ones(n)])
#     c = np.array([target_return, 1.0])
#     b = np.zeros(n)

#     x_star = projected_conjugate_gradient(sigma, c, A, b, tol=tol)
#     return x_star, mu


# def display_portfolio(
#     tickers: Sequence[str], weights: np.ndarray, mu: np.ndarray
# ) -> None:
#     """Print weights, sum-to-one check, and achieved return."""
#     print("Portfolio weights:")
#     for ticker, w in zip(tickers, weights):
#         print(f"  {ticker:>5}: {w: .4f}")
#     print(f"  Sum of weights : {weights.sum(): .6f}")
#     print(f"  Achieved return: {float(mu @ weights): .6f}\n")


# def test_with_yfinance(
#     tickers: Sequence[str],
#     start: str = "2024-12-20",
#     end: str = date.today().isoformat(),
#     target_return: float = 0.10,
# ) -> None:
#     returns = fetch_stock_data(tickers, start, end)
#     print("Sample returns:")
#     print(returns.head(), "\n")

#     weights, mu = solve_markowitz(returns, target_return)
#     display_portfolio(tickers, weights, mu)


# def test_with_synthetic(
#     n: int = 5, target_factor: float = 0.8, tol: float = 1e-10
# ) -> None:
#     # Build a random SPD covariance
#     Q, _ = np.linalg.qr(np.random.randn(n, n))
#     Λ = np.linspace(1, 10, n)
#     Σ = Q @ np.diag(Λ) @ Q.T

#     mu = np.random.rand(n)
#     target_return = target_factor * mu.mean()

#     # Constraints
#     A = np.vstack([mu, np.ones(n)])
#     c = np.array([target_return, 1.0])
#     b = np.zeros(n)

#     weights = projected_conjugate_gradient(Σ, c, A, b, tol=tol)
#     tickers = [f"Asset{i+1}" for i in range(n)]
#     display_portfolio(tickers, weights, mu)


# def plot_cov_corr_heatmaps(returns: pd.DataFrame) -> None:
#     """
#     Plot covariance and correlation heatmaps of asset returns using seaborn.

#     Parameters
#     ----------
#     returns : pd.DataFrame
#         DataFrame of shape (T, n) containing asset returns, with columns as tickers.
#     """
#     cov = returns.cov()
#     corr = returns.corr()
    
#     # Covariance heatmap
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(cov, xticklabels=cov.columns, yticklabels=cov.columns, annot=True)
#     plt.title("Covariance Heatmap")
#     plt.tight_layout()
#     plt.show()

#     # Correlation heatmap
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True)
#     plt.title("Correlation Heatmap")
#     plt.tight_layout()
#     plt.show()



# def main() -> None:
#     tickers = [
#         "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
#         "META", "TSLA", "BRK-B", "UNH",  "JPM"
#     ]
#     test_with_yfinance(tickers)
#     # test_with_synthetic()



# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
from __future__ import annotations
from datetime import date
from typing import Sequence, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt

from projected_cg import projected_conjugate_gradient


def fetch_stock_data(
    tickers: Sequence[str],
    start: str = "2024-12-20",
    end: str = date.today().isoformat(),
    auto_adjust: bool = False,
) -> pd.DataFrame:
    """
    Download closing prices for `tickers` and return daily simple returns.
    Drops any ticker with missing data.
    """
    df = yf.download(
        tickers, start=start, end=end, progress=False, auto_adjust=auto_adjust
    )["Close"]
    df = df.dropna(axis=1)
    return df.pct_change().dropna()


def plot_cov_corr_heatmaps(returns: pd.DataFrame) -> None:
    """
    Plot covariance and correlation heatmaps of asset returns using seaborn.

    Parameters
    ----------
    returns : pd.DataFrame
        DataFrame of shape (T, n) containing asset returns, with columns as tickers.
    """
    cov = returns.cov()
    corr = returns.corr()
    
    # Covariance heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cov, xticklabels=cov.columns, yticklabels=cov.columns, annot=True)
    plt.title("Covariance Heatmap")
    plt.tight_layout()
    plt.show()

    # Correlation heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()


def solve_markowitz(
    returns: pd.DataFrame, target_return: float, tol: float = 1e-8
) -> Tuple[np.ndarray, np.ndarray]:

    mu = returns.mean().values
    sigma = returns.cov().values
    n = mu.size

    # Build constraints A x = c
    A = np.vstack([mu, np.ones(n)])
    c = np.array([target_return, 1.0])
    b = np.zeros(n)

    x_star = projected_conjugate_gradient(sigma, c, A, b, tol=tol)
    return x_star, mu


def display_portfolio(
    tickers: Sequence[str], weights: np.ndarray, mu: np.ndarray
) -> None:
    """Print weights, sum-to-one check, and achieved return."""
    print("Portfolio weights:")
    for ticker, w in zip(tickers, weights):
        print(f"  {ticker:>5}: {w: .4f}")
    print(f"  Sum of weights : {weights.sum(): .6f}")
    print(f"  Achieved return: {float(mu @ weights): .6f}\n")


def test_with_yfinance(
    tickers: Sequence[str],
    start: str = "2024-12-20",
    end: str = date.today().isoformat(),
    target_return: float = 0.10,
) -> None:
    returns = fetch_stock_data(tickers, start, end)
    print("Sample returns:")
    print(returns.head(), "\n")

    # Plot covariance and correlation heatmaps
    plot_cov_corr_heatmaps(returns)

    weights, mu = solve_markowitz(returns, target_return)
    display_portfolio(tickers, weights, mu)


def test_with_synthetic(
    n: int = 5, target_factor: float = 0.8, tol: float = 1e-10
) -> None:
    # Build a random SPD covariance
    Q, _ = np.linalg.qr(np.random.randn(n, n))
    Λ = np.linspace(1, 10, n)
    Σ = Q @ np.diag(Λ) @ Q.T

    mu = np.random.rand(n)
    target_return = target_factor * mu.mean()

    # Constraints
    A = np.vstack([mu, np.ones(n)])
    c = np.array([target_return, 1.0])
    b = np.zeros(n)

    weights = projected_conjugate_gradient(Σ, c, A, b, tol=tol)
    tickers = [f"Asset{i+1}" for i in range(n)]
    display_portfolio(tickers, weights, mu)


def main() -> None:
    tickers = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
        "META", "TSLA", "BRK-B", "UNH",  "JPM"
    ]
    test_with_yfinance(tickers)
    # test_with_synthetic()


if __name__ == "__main__":
    main()
