import os
import sys
import time

# 1) Make project root importable
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)


import numpy as np
import matplotlib.pyplot as plt
from models.conjugate_gradient import conjugate_gradient

def cg_residuals(A, b, tol=1e-10, maxiter=None):
    x, its, _, xstar = conjugate_gradient(A, b, np.zeros_like(b),
                                        tolerance=tol,
                                        max_iterations=maxiter)
    return [np.linalg.norm(b - A.dot(xk)) for xk in its]

# Ensure results directory exists
results_dir = os.path.join(project_root, "results", "spd_nonsymmetric")
os.makedirs(results_dir, exist_ok=True)

# Generate a non-symmetric matrix with controlled condition number
n = 200
rng = np.random.default_rng(42)

# Create base SPD matrix
D = np.diag(np.logspace(0, 3, n))  # Eigenvalues from 1 to 10^3
Q = np.linalg.qr(rng.standard_normal((n, n)))[0]
A_spd = Q @ D @ Q.T

# Add skew-symmetric part to make it non-symmetric
B = rng.standard_normal((n, n))
B = (B - B.T) * 0.1  # Small skew-symmetric perturbation
A = A_spd + B  # A is now non-symmetric but still positive definite

# Random right-hand side
b = rng.standard_normal(n)
b = b / np.linalg.norm(b)

# Normal equations
ATA = A.T @ A
ATb = A.T @ b

# Run CG on normal equations
res_normal = cg_residuals(ATA, ATb)

# Run CG on original system (now SPD)
res_orig = cg_residuals(A_spd, b)

# Plot results
plt.figure(figsize=(10, 6))
plt.loglog(
    np.arange(1, len(res_normal) + 1),  # Start from 1 to avoid log(0)
    res_normal, 
    marker='o', 
    markersize=4, 
    label='Normal Equations'
)
plt.loglog(
    np.arange(1, len(res_orig) + 1),  # Start from 1 to avoid log(0)
    res_orig, 
    marker='s', 
    markersize=4, 
    label='Original SPD'
)
plt.xlabel('Iteration k')
plt.ylabel('Residual ‖b−Axₖ‖₂')
plt.title('CG Convergence: Normal Equations vs Original SPD (n={})'.format(n))
plt.legend()
plt.grid(which='both', linestyle='--', alpha=0.5)
plt.savefig(os.path.join(results_dir, 'cg_nonsymmetric.png'), dpi=300, bbox_inches='tight') 