import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm, eigvalsh
from pathlib import Path
import sys
from sklearn.decomposition import PCA

# Add project root to path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from resources.generate_spd import make_spd_matrix
from models.steepest_descent import steepest_descent

def project_to_2d(residuals):
    """Project residuals to 2D using PCA to best show orthogonality."""
    # Stack residuals into a matrix
    R = np.column_stack(residuals)
    # Fit PCA
    pca = PCA(n_components=2)
    # Project data
    R_2d = pca.fit_transform(R.T).T
    return R_2d, pca.components_

def plot_residuals_with_contours(A, b, residuals, n, k, ax):
    """
    Plots residual vectors with contour lines of the quadratic function.
    """
    # Project residuals to 2D
    R_2d, V = project_to_2d(residuals[:2])  # Only first two residuals
    
    # Project matrix A to the same 2D space
    A_2d = V @ A @ V.T
    b_2d = V @ b
    
    # Create grid of points
    x = np.linspace(-6, 2, 100)
    y = np.linspace(-2, 8, 100)
    X, Y = np.meshgrid(x, y)
    
    # Compute quadratic function values in projected space
    Z = np.zeros_like(X)
    for i in range(len(x)):
        for j in range(len(y)):
            point = np.array([X[i,j], Y[i,j]])
            Z[i,j] = 0.5 * point.dot(A_2d.dot(point)) - b_2d.dot(point)
    
    # Plot contour lines
    ax.contour(X, Y, Z.T, levels=20, colors='gray', linestyles='-', alpha=0.5)
    
    # Plot projected residuals
    r0 = R_2d[:,0]
    r1 = R_2d[:,1]
    
    # Plot vectors with arrows
    ax.quiver(0, 0, r0[0], r0[1],
              angles='xy', scale_units='xy', scale=1,
              color='black', width=0.01)
    ax.quiver(0, 0, r1[0], r1[1],
              angles='xy', scale_units='xy', scale=1,
              color='black', width=0.01)
    
    # Compute angle between residuals in original space
    cos_theta = residuals[0].dot(residuals[1]) / (norm(residuals[0]) * norm(residuals[1]))
    angle = np.degrees(np.arccos(np.clip(cos_theta, -1, 1)))
    
    # Set equal aspect ratio and grid
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Add labels
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_title(f'n={n}, κ={k}\nθ = {angle:.2f}°')

def main():
    np.random.seed(42)
    
    # Test different problem sizes and condition numbers
    ns = [10, 100, 1000]
    kappas = [10, 100, 1000]
    
    # Create subplot grid
    fig, axes = plt.subplots(len(ns), len(kappas), figsize=(15, 15))
    plt.subplots_adjust(hspace=0.4, wspace=0.4)
    
    for i, n in enumerate(ns):
        for j, k in enumerate(kappas):
            print(f"\nTesting n={n}, κ={k}")
            
            # Generate SPD matrix and RHS
            A = make_spd_matrix(n_dim=n, condition_number=k,
                              distribution='linear', random_state=42)
            b = np.random.randn(n)
            b /= norm(b)
            x0 = np.zeros(n)
            
            # Run steepest descent
            x_final, history, iters, x_star = steepest_descent(
                A, b, x0,
                tolerance=np.finfo(float).eps,
                max_iterations=2,
                store_history=True
            )
            
            # Compute residuals
            residuals = []
            for x in history:
                r = b - A @ x
                residuals.append(r)
            
            # Check orthogonality
            r0, r1 = residuals[0], residuals[1]
            ip = float(r0.dot(r1))
            cos_theta = ip / (norm(r0) * norm(r1))
            angle = np.degrees(np.arccos(np.clip(cos_theta, -1, 1)))
            print(f"‖r₀‖ = {norm(r0):.3e}")
            print(f"‖r₁‖ = {norm(r1):.3e}")
            print(f"⟨r₀,r₁⟩ = {ip:.3e}")
            print(f"cos(θ) = {cos_theta:.3e}")
            print(f"θ = {angle:.2f}°")
            
            # Plot in appropriate subplot
            plot_residuals_with_contours(A, b, residuals, n, k, axes[i,j])
    
    plt.suptitle('Steepest Descent Residuals\nProjected to 2D to Show Orthogonality', 
                 fontsize=16, y=1.02)
    plt.savefig("residuals_all_cases.png", dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main() 