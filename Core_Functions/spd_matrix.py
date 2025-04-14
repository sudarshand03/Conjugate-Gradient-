from typing import List, Optional
import numpy as np

def generate_spd_matrix(n: int, condition_number: float = 10.0, eigenvalues: Optional[List[float]] = None) -> np.ndarray:
    """
    Generate a symmetric positive-definite matrix using Q^T D Q decomposition.

    Args:
        n (int): Size of the square matrix.
        condition_number (float): Desired condition number (max/min eigenvalue ratio).
        eigenvalues (Optional[List[float]]): Custom eigenvalues (if provided).

    Returns:
        np.ndarray: SPD matrix A of shape (n, n).
    """
    # Generate random matrix and orthogonalize via QR
    H: np.ndarray = np.random.randn(n, n)
    Q: np.ndarray
    _, Q = np.linalg.qr(H)

    # Define eigenvalues: default linspace or custom
    if eigenvalues is None:
        eig_vals: np.ndarray = np.linspace(1, condition_number, n)
    else:
        # Ensure eigenvalues are positive and exactly n in length
        eig_vals = np.array(eigenvalues, dtype=np.float64)
        if len(eig_vals) != n:
            # Repeat eigenvalues to match n exactly
            eig_vals = np.tile(eig_vals, (n // len(eig_vals)) + 1)[:n]
        if np.any(eig_vals <= 0):
            raise ValueError("Eigenvalues must be positive.")
    
    # Create diagonal matrix and compute A
    D: np.ndarray = np.diag(eig_vals)
    A: np.ndarray = Q.T @ D @ Q
    return A