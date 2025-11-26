"""
Covariance Matrix Generators
============================

This module provides a suite of functions for generating synthetic covariance
matrices under a wide variety of structural assumptions. These generators are
useful for simulation studies, benchmarking hypothesis tests, and exploring
properties of covariance estimators in high-dimensional settings.

Implemented generators include:

- Random correlation-based covariances with tunable off-diagonal strength
  (`sample_cov_uniform_correlation`).
- Covariance matrices derived from Gaussian Orthogonal Ensemble (GOE) samples
  (`sample_goe`).
- Random spectral decompositions with controlled eigenvalue distributions
  (`generate_spectral_cov`).
- Toeplitz structures mimicking AR(1)-like correlations
  (`generate_toeplitz_cov`).
- Block-diagonal matrices with independent positive semi-definite blocks
  (`generate_block_diagonal_cov`).
- Low-rank factor models plus isotropic Gaussian noise
  (`generate_low_rank_cov`).
- Covariances obtained by inverting sparse precision matrices
  (`generate_sparse_precision_cov`).
- Wishart/Marčenko–Pastur ensemble covariances formed from i.i.d. Gaussian data
  (`generate_marchenko_pastur`).
- Spiked covariance models with one or more inflated eigenvalues
  (`generate_spiked_covariance`).

Each generator ensures the resulting matrix is symmetric positive definite
(or adjusted with minimal jitter when needed). All functions accept a NumPy
random number generator (`rng`) to allow reproducible simulations.

Typical usage
-------------
>>> from covtest.simulate.generate_covariances import generate_toeplitz_cov
>>> Sigma = generate_toeplitz_cov(p=50, rho=0.7)
"""

import numpy as np
from scipy.stats import ortho_group


def sample_goe(p, rng=None):
    """
    Sample a p x p matrix from the Gaussian Orthogonal Ensemble (GOE).

    Parameters
    ----------
    p : int
        The size of the matrix.
    rng : np.random.Generator
        A NumPy-compatible random number generator.

    Returns
    -------
    A : np.ndarray
        A p x p symmetric matrix from the GOE.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Draw the upper triangle of the matrix, including the diagonal
    A = rng.normal(loc=0.0, scale=1.0, size=(p, p))
    A = (A + A.T) / np.sqrt(2.0)  # ensure symmetry and correct scaling
    return A


def sample_cov_uniform_correlation(K, level="low", rng=None):
    """
    Construct a KxK covariance matrix R with unit variances and
    off-diagonal correlations drawn uniformly within a range, as
    in the paper.

    level in {"low","medium","high"} sets the Uniform ranges:
      low:    U(0, 0.3)
      medium: U(0.3, 0.5)
      high:   U(0.5, 0.7)
    """
    if rng is None:
        rng = np.random.default_rng()

    ranges = {
        "low": (0.0, 0.3),
        "medium": (0.3, 0.5),
        "high": (0.5, 0.7),
    }
    if level not in ranges:
        raise ValueError("level must be one of {'low','medium','high'}")

    lo, hi = ranges[level]
    R = np.eye(K)
    # Random upper triangle, symmetric fill
    for i in range(K):
        for j in range(i + 1, K):
            r = rng.uniform(lo, hi)
            R[i, j] = r
            R[j, i] = r

    # Ensure positive-definite (jitter if needed)
    # Add a tiny diagonal if eigenvalues are too small
    min_eig = np.linalg.eigvalsh(R).min()
    if min_eig < 1e-6:
        R += (1e-6 - min_eig + 1e-8) * np.eye(K)
    return R


# 2. Spectral Decomposition
def generate_spectral_cov(p, rng=None):
    """
    Generate a random covariance matrix by sampling eigenvalues and a
    random orthogonal basis.

    Parameters:
        p (int): Dimensionality of the covariance matrix.
        rng (np.random.Generator): Random number generator.

    Returns:
        np.ndarray: A (p x p) positive definite covariance matrix with controlled spectrum.
    """
    if rng is None:
        rng = np.random.default_rng()

    eigenvalues = rng.uniform(0.5, 2.0, size=p)
    Q = ortho_group.rvs(dim=p, random_state=rng)
    return Q @ np.diag(eigenvalues) @ Q.T


# 3. Banded Toeplitz (AR(1))
def generate_toeplitz_cov(p, rho):
    """
    Generate a Toeplitz covariance matrix assuming an AR(1) process.

    Parameters:
        p (int): Dimensionality of the covariance matrix.
        rho (float): Autoregressive correlation coefficient, must be
        in (-1, 1).

    Returns:
        np.ndarray: A (p x p) symmetric Toeplitz matrix with
        exponentially decaying correlations.
    """
    return np.fromfunction(lambda i, j: rho ** np.abs(i - j), (p, p), dtype=int)


# 4. Block Diagonal
def generate_block_diagonal_cov(p, block_size, rng=None):
    """
    Generate a block diagonal covariance matrix with each block
    being positive semi-definite.

    Parameters:
        p (int): Total dimensionality (must be divisible by block_size).
        block_size (int): Size of each individual block.
        rng (np.random.Generator): Random number generator.

    Returns:
        np.ndarray: A (p x p) block diagonal covariance matrix with
        block-structured dependencies.
    """
    if rng is None:
        rng = np.random.default_rng()

    blocks = []
    for _ in range(p // block_size):
        B = rng.normal(size=(block_size, block_size))
        block = B @ B.T
        blocks.append(block)
    return np.block(
        [
            [
                blocks[i] if i == j else np.zeros_like(blocks[0])
                for j in range(len(blocks))
            ]
            for i in range(len(blocks))
        ]
    )


# 5. Low-Rank + Noise
def generate_low_rank_cov(p, rank, noise_var, rng=None):
    """
    Generate a covariance matrix with low-rank latent structure plus
    isotropic Gaussian noise.

    Parameters:
        p (int): Dimensionality of the covariance matrix.
        rank (int): Rank of the latent structure (number of latent factors).
        noise_var (float): Variance of isotropic Gaussian noise.
        rng (np.random.Generator): Random number generator.

    Returns:
        np.ndarray: A (p x p) positive definite covariance matrix
        with low-rank plus noise structure.
    """
    if rng is None:
        rng = np.random.default_rng()

    B = rng.normal(size=(p, rank))
    return B @ B.T + noise_var * np.eye(p)


# 6. Sparse Precision Matrix (invert to get covariance)
def generate_sparse_precision_cov(p, sparsity=0.2, rng=None):
    """
    Generate a sparse precision matrix and return its inverse as a
    covariance matrix.

    Parameters:
        p (int): Dimensionality of the precision/covariance matrix.
        sparsity (float): Probability of off-diagonal entries being
            non-zero (between 0 and 1).
        rng (np.random.Generator): Random number generator.

    Returns:
        np.ndarray: A (p x p) symmetric positive definite covariance
        matrix derived from a sparse precision matrix.
    """
    if rng is None:
        rng = np.random.default_rng()

    prec = np.eye(p)
    for i in range(p):
        for j in range(i + 1, p):
            if rng.random() < sparsity:
                val = rng.uniform(-0.5, 0.5)
                prec[i, j] = prec[j, i] = val
    cov = np.linalg.inv(prec)
    return (cov + cov.T) / 2  # ensure symmetry


# 8. Marcenko-Pastur (White Wishart)
def generate_marchenko_pastur(p, n, rng=None):
    """
    Generate a covariance matrix from the white Wishart ensemble, useful
    for studying Marčenko–Pastur law.

    Parameters:
        p (int): Dimensionality of the covariance matrix.
        n (int): Number of independent samples (controls the aspect ratio).
        rng (np.random.Generator): Random number generator.

    Returns:
        np.ndarray: A (p x p) empirical covariance matrix formed from
        standard normal samples.
    """
    if rng is None:
        rng = np.random.default_rng()

    X = rng.normal(size=(n, p))
    return X.T @ X / n


def generate_spiked_covariance(
    p, spike_eigenvalue=10.0, num_spikes=1, base_variance=1.0, rng=None
):
    """
    Generate a covariance matrix with one or more large eigenvalue "spikes".

    Parameters:
        p (int): Dimensionality of the covariance matrix.
        spike_eigenvalue (float): Value of the spiked eigenvalues
                                (>> base_variance).
        num_spikes (int): Number of leading eigenvalues to spike
                                (must be <= p).
        base_variance (float): Variance of non-spiked components
                                (default = 1.0).
        rng (np.random.Generator or None): Optional random number generator.

    Returns:
        np.ndarray: (p x p) symmetric positive definite covariance matrix
        with specified spike structure.
    """
    if rng is None:
        rng = np.random.default_rng()
    if num_spikes > p:
        raise ValueError("Number of spikes cannot exceed dimension p.")

    # Construct eigenvalues: leading values are large (spikes), rest are base
    eigenvalues = np.full(p, base_variance)
    eigenvalues[:num_spikes] = spike_eigenvalue

    # Random orthogonal matrix Q via QR decomposition
    A = rng.normal(size=(p, p))
    Q, _ = np.linalg.qr(A)

    # Construct covariance: Q Λ Qᵗ
    cov = Q @ np.diag(eigenvalues) @ Q.T
    return cov
