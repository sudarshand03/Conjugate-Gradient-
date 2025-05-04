#!/usr/bin/env python3
"""
generate_spd.py

Generate random symmetric positive-definite (SPD) matrices with
various spectral specifications.
"""

from typing import Optional, Sequence, Union
import numpy as np
from numpy.random import Generator, default_rng


# def make_spd_matrix(
#     n_dim: int,
#     *,
#     condition_number: Optional[float] = None,
#     eigenvalues: Optional[Sequence[float]] = None,
#     cluster_centers: Optional[Sequence[float]] = None,
#     cluster_sizes: Optional[Sequence[int]] = None,
#     cluster_std: float = 0.01,
#     distribution: str = "linear",
#     random_state: Optional[Union[int, Generator]] = None
# ) -> np.ndarray:
#     """
#     Generate an SPD matrix by specifying exactly one of:
#       - eigenvalues: explicit list of positive eigenvalues (tiled/truncated)
#       - cluster_centers & cluster_sizes: Gaussian clusters spectrum
#       - condition_number: uniform or log-uniform in [1, κ]
#     If none specified, falls back to Wishart-style M M^T + n I.
#     """
#     rng = default_rng(random_state) if not isinstance(random_state, Generator) else random_state

#     # Build the desired spectrum
#     if eigenvalues is not None:
#         vals = np.array(eigenvalues, dtype=float)
#         if np.any(vals <= 0):
#             raise ValueError("All eigenvalues must be positive.")
#         reps = int(np.ceil(n_dim / vals.size))
#         vals = np.tile(vals, reps)[:n_dim]

#     elif cluster_centers is not None:
#         if cluster_sizes is None or len(cluster_sizes) != len(cluster_centers):
#             raise ValueError("cluster_sizes must match cluster_centers length.")
#         if sum(cluster_sizes) > n_dim:
#             raise ValueError("Sum of cluster_sizes cannot exceed n_dim.")
#         vals_list = []
#         for center, size in zip(cluster_centers, cluster_sizes):
#             samples = rng.standard_normal(size) * cluster_std + center
#             samples = np.clip(samples, 1e-8, None)
#             vals_list.append(samples)
#         remainder = n_dim - sum(cluster_sizes)
#         if remainder > 0:
#             vals_list.append(np.ones(remainder))
#         vals = np.concatenate(vals_list)

#     elif condition_number is not None:
#         if distribution == "linear":
#             vals = np.linspace(1.0, condition_number, n_dim)
#         elif distribution == "log":
#             vals = np.logspace(0.0, np.log10(condition_number), n_dim)
#         else:
#             raise ValueError("distribution must be 'linear' or 'log'.")
#     else:
#         # Fallback: Wishart-style SPD
#         M = rng.standard_normal((n_dim, n_dim))
#         return M @ M.T + n_dim * np.eye(n_dim)

#     # Orthonormal basis via QR
#     H = rng.standard_normal((n_dim, n_dim))
#     Q, _ = np.linalg.qr(H)

#     # Assemble SPD matrix
#     return Q @ np.diag(vals) @ Q.T


def generate_spd(
    n_dim: int,
    condition_number: float = 10.0,
    distribution: str = "log",
    random_state: Optional[Union[int, Generator]] = None
) -> np.ndarray:
    """
    Generate an SPD matrix with eigenvalues in [1, κ], linear or log spacing.
    """
    rng = default_rng(random_state)
    if distribution == "log":
        vals = np.logspace(0.0, np.log10(condition_number), n_dim)
    else:
        vals = np.linspace(1.0, condition_number, n_dim)
    H = rng.standard_normal((n_dim, n_dim))
    Q, _ = np.linalg.qr(H)
    return Q @ np.diag(vals) @ Q.T


def generate_spd_distinct5(
    n_dim: int,
    eigenvalues: Optional[Sequence[float]] = None,
    random_state: Optional[Union[int, Generator]] = None
) -> np.ndarray:
    """
    Generate an SPD matrix with exactly 5 distinct eigenvalues,
    tiled/truncated to length n_dim.
    """
    rng = default_rng(random_state)
    if eigenvalues is None:
        base_vals = np.arange(1, 6, dtype=float)
    else:
        base_vals = np.array(eigenvalues, dtype=float)
        if base_vals.size != 5:
            raise ValueError("eigenvalues must have length 5.")
    if np.any(base_vals <= 0):
        raise ValueError("All eigenvalues must be positive.")
    reps = int(np.ceil(n_dim / 5))
    vals = np.tile(base_vals, reps)[:n_dim]
    H = rng.standard_normal((n_dim, n_dim))
    Q, _ = np.linalg.qr(H)
    return Q @ np.diag(vals) @ Q.T


def generate_spd_5clusters(
    n_dim: int,
    low: float = 1.0,
    high: float = 5.0,
    cluster_std: float = 0.01,
    random_state: Optional[Union[int, Generator]] = None
) -> np.ndarray:
    """
    Generate an SPD matrix with 5 clusters of eigenvalues between low and high.
    """
    rng = default_rng(random_state)
    centers = np.linspace(low, high, 5)
    base = n_dim // 5
    sizes = [base + (1 if i < n_dim % 5 else 0) for i in range(5)]
    vals_list = []
    for c, sz in zip(centers, sizes):
        samples = rng.standard_normal(sz) * cluster_std + c
        samples = np.clip(samples, 1e-8, None)
        vals_list.append(samples)
    vals = np.concatenate(vals_list)
    H = rng.standard_normal((n_dim, n_dim))
    Q, _ = np.linalg.qr(H)
    return Q @ np.diag(vals) @ Q.T
