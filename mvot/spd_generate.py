from typing import List, Optional
import numpy as np

def generate_spd_matrix(n: int, condition_number: float = 10.0, eigenvalues: Optional[List[float]] = None,
                        perturb: bool = False, epsilon: float = 0.0) -> np.ndarray:
    """
    Generate a symmetric positive-definite (SPD) matrix using Q^T D Q decomposition.

    Args:
        n (int): Size of the square matrix (n x n).
        condition_number (float): Desired condition number (λ_max / λ_min), default 10.0.
        eigenvalues (Optional[List[float]]): Custom positive eigenvalues; if None, uses linspace.
        perturb (bool): If True, perturb eigenvalues with small random noise.
        epsilon (float): If >0, add perturbation to break symmetry (A becomes non-symmetric).

    Returns:
        np.ndarray: Matrix A of shape (n, n); SPD unless epsilon > 0.

    Raises:
        ValueError: If eigenvalues are non-positive.
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    # Generate orthogonal matrix Q via QR decomposition
    H: np.ndarray = np.random.randn(n, n)
    _, Q = np.linalg.qr(H)

    # Generate eigenvalues
    if eigenvalues is None:
        eig_vals = np.linspace(1, condition_number, n)  # Default: linear spacing from 1 to κ
    else:
        eig_vals = np.array(eigenvalues, dtype=np.float64)
        if perturb:
            # Add small random perturbation to eigenvalues
            eig_vals += 0.01 * np.random.randn(len(eig_vals))
        # Tile eigenvalues if fewer than n, truncate if more
        if len(eig_vals) != n:
            eig_vals = np.tile(eig_vals, (n // len(eig_vals) + 1))[:n]
        # Ensure all eigenvalues are positive
        if np.any(eig_vals <= 0):
            raise ValueError("All eigenvalues must be positive for SPD property.")

    # Construct diagonal matrix D and form A = Q^T D Q
    D = np.diag(eig_vals)
    A = Q.T @ D @ Q

    # Optionally break symmetry
    if epsilon > 0:
        A += epsilon * np.random.randn(n, n)  # Add random perturbation

    return A