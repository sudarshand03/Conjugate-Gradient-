# generate_spd.py

from typing import (
    Optional,
    Sequence,
    Union,
    Tuple,
)
import numpy as np
from numpy.random import Generator, default_rng
from numpy import ndarray


def make_spd_matrix(
    n_dim: int,
    *,
    condition_number: Optional[float] = None,
    eigenvalues: Optional[Sequence[float]] = None,
    cluster_centers: Optional[Sequence[float]] = None,
    cluster_sizes: Optional[Sequence[int]] = None,
    cluster_std: float = 0.01,
    distribution: str = "linear",
    random_state: Optional[Union[int, Generator]] = None
) -> ndarray:
    """
    Generate a random symmetric positive-definite matrix with flexible spectrum.

    You can specify *one* of:
      - `eigenvalues`: an explicit list of positive eigenvalues,
      - `cluster_centers` + `cluster_sizes`: a mixture-of-Gaussians spectrum,
      - `condition_number`: generate uniform (or log-uniform) eigenvalues in [1, κ].

    Parameters
    ----------
    n_dim : int
        Dimension of the SPD matrix (n_dim x n_dim).
    condition_number : float, optional
        Desired condition number  If set (and `eigenvalues` is None),
        generates `n_dim` eigenvalues in [1, κ] via `distribution`.
    eigenvalues : sequence of float, optional
        Exact positive eigenvalues to use (will be tiled/truncated to length n_dim).
    cluster_centers : sequence of float, optional
        Centers of Gaussian clusters for eigenvalues.
    cluster_sizes : sequence of int, optional
        Sizes of each cluster (must sum to ≤ n_dim).
    cluster_std : float
        Standard deviation for sampling around each cluster center.
    distribution : {'linear', 'log'}, default='linear'
        If `condition_number` is set, choose uniform or log-uniform spacing.
    random_state : int or Generator, optional
        Seed or RNG for reproducibility.

    Returns
    -------
    A : ndarray, shape (n_dim, n_dim)
        Random SPD matrix with the specified eigenvalue spectrum.

    Raises
    ------
    ValueError
        If inputs are inconsistent (e.g. cluster_centers without sizes).
    """
    # 1) Set up RNG
    rng: Generator = (
        default_rng(random_state)
        if not isinstance(random_state, Generator)
        else random_state
    )

    # 2) Build eigenvalue array
    if eigenvalues is not None:
        # Use the user list, tile/truncate to length n_dim
        vals = np.array(eigenvalues, dtype=float)
        if vals.min() <= 0:
            raise ValueError("All eigenvalues must be positive.")
        reps = int(np.ceil(n_dim / vals.size))
        vals = np.tile(vals, reps)[:n_dim]

    elif cluster_centers is not None:
        # Must have sizes for each cluster
        if cluster_sizes is None or len(cluster_sizes) != len(cluster_centers):
            raise ValueError(
                "cluster_sizes must match cluster_centers length.")
        if sum(cluster_sizes) > n_dim:
            raise ValueError("Sum of cluster_sizes cannot exceed n_dim.")
        # Sample each cluster
        vals_list = []
        for center, size in zip(cluster_centers, cluster_sizes):
            samples = rng.standard_normal(size) * cluster_std + center
            if np.any(samples <= 0):
                # clip to small positive to keep SPD
                samples = np.clip(samples, 1e-8, None)
            vals_list.append(samples)
        # Fill up with small eigenvalues = 1.0 if needed
        remainder = n_dim - sum(cluster_sizes)
        if remainder > 0:
            vals_list.append(np.ones(remainder))
        vals = np.concatenate(vals_list)

    elif condition_number is not None:
        # Uniformly or log-uniformly spaced spectrum in [1, κ]
        if distribution == "linear":
            vals = np.linspace(1.0, condition_number, n_dim)
        elif distribution == "log":
            vals = np.logspace(0.0, np.log10(condition_number), n_dim)
        else:
            raise ValueError("distribution must be 'linear' or 'log'.")
    else:
        # Default fallback: Wishart + shift (like sklearn)
        M = rng.standard_normal((n_dim, n_dim))
        A = M @ M.T + n_dim * np.eye(n_dim)
        return A

    # 3) Construct orthonormal basis Q via QR
    H = rng.standard_normal((n_dim, n_dim))
    _, Q = np.linalg.qr(H)

    # 4) Assemble A = Qᵀ (diag(vals)) Q
    D = np.diag(vals)
    A = Q.T @ D @ Q

    return A

# resources/generate_spd.py
# … (rest of your file unchanged) …


def generate_spd(
    n_dim: int,
    condition_number: float = 10.0,
    distribution: str = "linear",
    random_state: Optional[Union[int, Generator]] = None
) -> ndarray:
    """
    Generate an n_dim x n_dim SPD matrix with a specified condition number.
    """
    rng = default_rng(random_state)

    # 1) Create eigenvalue array
    if distribution == "log":
        vals = np.logspace(0.0, np.log10(condition_number), n_dim)
    else:
        vals = np.linspace(1.0, condition_number, n_dim)

    # 2) Generate random orthonormal Q via QR
    H = rng.standard_normal((n_dim, n_dim))
    Q, _ = np.linalg.qr(H)

    # 3) Form A = Q.T @ diag(vals) @ Q
    return Q.T @ np.diag(vals) @ Q


def generate_spd_distinct5(
    n_dim: int,
    eigenvalues: Optional[Sequence[float]] = None,
    random_state: Optional[Union[int, Generator]] = None
) -> ndarray:
    """
    Generate an SPD matrix with exactly 5 distinct eigenvalues.
    """
    rng = default_rng(random_state)

    # 1) Determine the 5 base values
    if eigenvalues is None:
        base_vals = np.array([1, 2, 3, 4, 5], dtype=float)
    else:
        base_vals = np.array(eigenvalues, dtype=float)
        if base_vals.size != 5:
            raise ValueError("eigenvalues must have length 5")

    if np.any(base_vals <= 0):
        raise ValueError("All eigenvalues must be positive")

    # 2) Tile/truncate to length n_dim
    reps = int(np.ceil(n_dim / 5))
    vals = np.tile(base_vals, reps)[:n_dim]

    # 3) Build random orthonormal Q
    H = rng.standard_normal((n_dim, n_dim))
    Q, _ = np.linalg.qr(H)

    return Q.T @ np.diag(vals) @ Q


def generate_spd_5clusters(
    n_dim: int,
    low: float = 1.0,
    high: float = 5.0,
    cluster_std: float = 0.01,
    random_state: Optional[Union[int, Generator]] = None
) -> ndarray:
    """
    Generate an SPD matrix with 5 clusters of eigenvalues.
    """
    rng = default_rng(random_state)

    # 1) Define 5 cluster centers uniformly in [low, high]
    centers = np.linspace(low, high, 5)

    # 2) Compute cluster sizes (nearly equal)
    base = n_dim // 5
    sizes = [base] * 5
    for i in range(n_dim % 5):
        sizes[i] += 1

    # 3) Sample around each center
    vals_list = []
    for c, sz in zip(centers, sizes):
        samples = rng.standard_normal(sz) * cluster_std + c
        samples = np.clip(samples, 1e-8, None)
        vals_list.append(samples)
    vals = np.concatenate(vals_list)

    # 4) Build random orthonormal Q
    H = rng.standard_normal((n_dim, n_dim))
    Q, _ = np.linalg.qr(H)

    return Q.T @ np.diag(vals) @ Q

if __name__ == "__main__":
    import numpy as np

    # 1) Uniform spacing, κ=100
    A1 = make_spd_matrix(50, condition_number=100.0,
                         distribution="linear", random_state=0)
    print("Eigenvalues (linear):", np.linalg.eigvalsh(
        A1)[:5], "…", np.linalg.eigvalsh(A1)[-5:])

    # 2) Log-uniform spacing, κ=1e6
    A2 = make_spd_matrix(50, condition_number=1e6,
                         distribution="log", random_state=0)
    print("Eigenvalues (log):", np.linalg.eigvalsh(
        A2)[:5], "…", np.linalg.eigvalsh(A2)[-5:])

    # 3) Custom clusters: two clusters around 1 and 100
    A3 = make_spd_matrix(
        60,
        cluster_centers=[1.0, 100.0],
        cluster_sizes=[30, 20],
        cluster_std=0.5,
        random_state=42
    )
    print("Eigenvalues (clustered):", np.linalg.eigvalsh(
        A3)[:5], "…", np.linalg.eigvalsh(A3)[-5:])

    # 4) Exact eigenvalue list
    A4 = make_spd_matrix(5, eigenvalues=[1, 2, 5], random_state=1)
    print("Eigenvalues (explicit):", np.linalg.eigvalsh(A4))
