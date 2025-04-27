from typing import Tuple, Optional
import numpy as np
from numpy import ndarray


def conjugate_gradient(
    matrix: ndarray,
    rhs: ndarray,
    initial_guess: ndarray,
    tolerance: float = 1e-8,
    max_iterations: Optional[int] = None,
    reorthogonalize: bool = True
) -> Tuple[ndarray, ndarray, int, ndarray]:
    """
    Solve Ax = b with the Conjugate Gradient method for SPD A,
    with optional re-orthogonalization and input validation.

    Args:
        matrix: SPD matrix or array of shape (n, n).
        rhs: Right-hand side vector of shape (n,).
        initial_guess: Initial solution vector of shape (n,).
        tolerance: Convergence tolerance for residual norm.
        max_iterations: Maximum CG steps (defaults to n).
        reorthogonalize: If True, apply Gram-Schmidt on search directions for stability.

    Returns:
        x: Approximate solution vector.
        iterates: Array of solution iterates, shape (k+1, n).
        iterations: Number of iterations performed.
        x_star: Exact solution via np.linalg.solve for reference.

    Raises:
        ValueError: If input dimensions mismatch or matrix is not SPD.
    """
    # Problem size
    n: int = rhs.size
    if max_iterations is None:
        max_iterations = n

    # Validate dimensions
    if matrix.shape != (n, n) or initial_guess.shape != (n,):
        raise ValueError(
            "Dimensions of matrix, rhs, and initial_guess must be (n,n) and (n,).")

    # Check SPD
    if not np.allclose(matrix, matrix.T, atol=1e-10):
        raise ValueError("Matrix must be symmetric.")
    eigenvalues = np.linalg.eigvalsh(matrix)
    if np.any(eigenvalues <= 0):
        raise ValueError("Matrix must be positive definite.")

    # Compute exact solution for reference
    x_star: ndarray = np.linalg.solve(matrix, rhs)

    # Pre-allocate iterates history
    iterates: ndarray = np.zeros((max_iterations + 1, n), dtype=matrix.dtype)
    x: ndarray = initial_guess.copy()
    iterates[0, :] = x

    # Initial residual and direction
    residual: ndarray = rhs - matrix.dot(x)
    direction: ndarray = residual.copy()
    resid_norm_sq: float = float(residual.dot(residual))

    if resid_norm_sq < tolerance**2:
        return x, iterates[:1, :], 0, x_star

    for iteration in range(1, max_iterations + 1):
        Ap: ndarray = matrix.dot(direction)
        denom: float = float(direction.dot(Ap))
        if denom <= 0:
            raise ValueError(
                "Matrix is not positive definite during iteration.")

        alpha: float = resid_norm_sq / denom
        x = x + alpha * direction
        iterates[iteration, :] = x

        # Update residual
        residual = residual - alpha * Ap
        new_resid_norm_sq: float = float(residual.dot(residual))

        # Convergence check
        if new_resid_norm_sq < tolerance**2:
            return x, iterates[: iteration + 1, :], iteration, x_star

        # Compute beta
        beta: float = new_resid_norm_sq / resid_norm_sq

        # Update direction
        new_direction = residual + beta * direction
        # Optional re-orthogonalization
        if reorthogonalize and iteration > 1:
            # Enforce A-orthogonality: subtract component along previous direction
            prev_Ap = matrix.dot(direction)
            coeff = float(new_direction.dot(prev_Ap) / denom)
            new_direction = new_direction - coeff * direction

        direction = new_direction
        resid_norm_sq = new_resid_norm_sq

    return x, iterates, max_iterations, x_star
