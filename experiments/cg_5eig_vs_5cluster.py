import numpy as np, matplotlib.pyplot as plt
from resources.generate_spd import make_spd_matrix
from models.conjugate_gradient import conjugate_gradient

def cg_residuals(A, b, tol=1e-10, maxiter=None):
    x, its, _, xstar = conjugate_gradient(A, b, np.zeros_like(b),
                                          tolerance=tol,
                                          max_iterations=maxiter)
    return [np.linalg.norm(b - A.dot(xk)) for xk in its]

n = 200
rng = np.random.default_rng(42)
b = rng.standard_normal(n); b /= np.linalg.norm(b)

# 5 distinct eigenvalues spread out
A1 = make_spd_matrix(n_dim=n, eigenvalues=[1, 10, 100, 1000, 10000], random_state=42)
res1 = cg_residuals(A1, b)

# 5 tight clusters with large gaps
centers = [1, 10, 100, 1000, 10000]
sizes = [n//5]*5
A2 = make_spd_matrix(n_dim=n,
                     cluster_centers=centers,
                     cluster_sizes=sizes,
                     cluster_std=0.001,  # Much tighter clusters
                     random_state=42)
res2 = cg_residuals(A2, b)

plt.figure(figsize=(10, 6))
plt.semilogy(res1, marker='o', markersize=4, label='5 distinct eigenvalues')
plt.semilogy(res2, marker='s', markersize=4, label='5 tight clusters')
plt.xlabel('Iteration k')
plt.ylabel('Residual ‖b−Axₖ‖₂')
plt.title(f'CG Convergence: Distinct vs Clustered Eigenvalues (n={n})')
plt.legend()
plt.grid(which='both', linestyle='--', alpha=0.5)
plt.savefig('results/cg_5eig_vs_5cluster.png', dpi=300, bbox_inches='tight')
