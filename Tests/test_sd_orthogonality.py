import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm, eigvalsh

def sd_residuals(A: np.ndarray, b: np.ndarray, x0: np.ndarray,
                 tol: float = 1e-10, max_iter: int = 10):
    """
    Runs Steepest Descent, returns the list of residuals [r0, r1, ..., r_k].
    """
    # Initial residual
    x = x0.copy()
    r = b - A.dot(x)
    residuals = [r.copy()]

    for k in range(1, max_iter+1):
        rr = float(r.dot(r))
        Ar = A.dot(r)
        denom = float(r.dot(Ar))
        if denom <= 0 or np.isnan(denom):
            break
        alpha = rr / denom

        # Update x and residual
        x = x + alpha * r
        r = b - A.dot(x)
        residuals.append(r.copy())

        if norm(r) < tol:
            break

    return residuals

def check_orthogonality(residuals):
    """
    Prints <r_{k-1}, r_k> for each k and shows it's (near) zero.
    """
    for k in range(1, len(residuals)):
        ip = float(residuals[k-1].dot(residuals[k]))
        print(f"<r_{k-1}, r_{k}> = {ip:.3e}")

def plot_residuals_2d(residuals, filename: str = None):
    """
    Draws the first few residual vectors in 2D using quiver.
    """
    plt.figure(figsize=(6,6))
    for k, r in enumerate(residuals[:3]):
        plt.quiver(0, 0, r[0], r[1],
                   angles='xy', scale_units='xy', scale=1,
                   label=f'r_{k}')
    plt.axhline(0, color='gray', linewidth=1)
    plt.axvline(0, color='gray', linewidth=1)
    plt.legend()
    plt.xlabel('x₁')
    plt.ylabel('x₂')
    plt.title('Steepest Descent Residuals (first 3)')
    plt.axis('equal')
    plt.grid(True)
    if filename:
        plt.savefig(filename, dpi=300)
    plt.show()

def main():
    # Example 2×2 SPD problem
    A = np.array([[4, 1],
                  [1, 3]], dtype=float)
    b = np.array([1, 2], dtype=float)
    x0 = np.zeros(2)

    # Sanity check: SPD?
    assert np.all(eigvalsh(A) > 0), "A must be SPD"

    # Compute residuals
    residuals = sd_residuals(A, b, x0, max_iter=3)

    # 1) Check orthogonality
    print("Inner products ⟨r_{k-1}, r_k⟩:")
    check_orthogonality(residuals)

    # 2) Plot first three residuals
    plot_residuals_2d(residuals)

if __name__ == "__main__":
    main()
