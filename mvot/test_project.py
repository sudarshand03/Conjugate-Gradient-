import numpy as np
import yfinance as yf
from datetime import date
from projected_cg import projected_conjugate_gradient

def fetch_stock_data(tickers, start="2024-12-20", end="2025-01-01"):
    # 1. Download closing prices
    data = yf.download(tickers, start=start, end=end)["Close"]
    # 2. Drop any tickers with missing data
    data = data.dropna(axis=1)
    # 3. Compute daily percentage change and drop the first NaN row
    returns = data.pct_change().dropna()
    return returns

def test_with_yfinance():
    # 1. Fetch returns for 4 stocks
    tickers = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "UNH", "JPM"
    ]
    
    R = fetch_stock_data(tickers)
    print(f"Returns: {R.head()}")
    # 2. Estimate covariance P and mean returns μ
    P = R.cov().values
    print(P)
    mu = R.mean().values
    # average_change=R.mean()
    n = len(mu)
    print(f"Average Percentage Change of Price per Stock: {mu}")
    print(f"Average Change of the Stock Vector: {mu.mean()}")

    # 3. Target return
    mu_p = 0.1

    # 4. Constraint A x = c
    A = np.vstack([mu, np.ones(n)])
    c = np.array([mu_p, 1.0])

    # 5. No linear term b=0
    b = np.zeros(n)

    # 6. Solve and display
    x_star = projected_conjugate_gradient(P, c, A, b, tol=1e-8)
    print("yfinance test weights:")
    for t, w in zip(tickers, x_star):
        print(f"  {t}: {w: .4f}")
    print("Sum of weights:", x_star.sum())
    print("Achieved return:", mu.dot(x_star))

def test_with_synthetic():
    # small synthetic SPD example: n=5
    n = 5
    Q, _ = np.linalg.qr(np.random.randn(n,n))
    eigenvalues = np.linspace(1,10,n)
    P_syn = Q @ np.diag(eigenvalues) @ Q.T

    mu_syn = np.random.rand(n)
    mu_p_syn = 0.8 * mu_syn.mean()

    A_syn = np.vstack([mu_syn, np.ones(n)])
    c_syn = np.array([mu_p_syn, 1.0])
    b_syn = np.zeros(n)

    x_syn = projected_conjugate_gradient(P_syn, c_syn, A_syn, b_syn, tol=1e-10)
    print("\nSynthetic test weights:")
    print(x_syn)
    print("Sum:", x_syn.sum(), "Return:", mu_syn.dot(x_syn))

if __name__ == "__main__":
    test_with_yfinance()
    # test_with_synthetic()
