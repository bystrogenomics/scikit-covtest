from math import erf, sqrt
from typing import Literal, Tuple

import numpy as np
from numpy.linalg import norm

ArrayLike = np.ndarray


def _phi(z: float) -> float:
    """Standard normal CDF without external dependencies."""
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


def spatial_median(
    X: np.ndarray, tol: float = 1e-7, max_iter: int = 500, eps: float = 1e-12,
) -> np.ndarray:
    """
    Compute the L1 (geometric) spatial median of points using Weiszfeld's algorithm.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Data matrix with rows as observations.
    tol : float, default 1e-7
        Convergence tolerance on relative parameter change.
    max_iter : int, default 500
        Maximum number of iterations.
    eps : float, default 1e-12
        Small constant to avoid division by zero.

    Returns
    -------
    mu : ndarray of shape (n_features,)
        Spatial median estimator.
    """
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    # Initialize with component-wise median which is robust and usually close
    mu = np.median(X, axis=0)

    for _ in range(max_iter):
        diff = X - mu
        d = np.maximum(norm(diff, axis=1), eps)
        w = 1.0 / d
        mu_new = (w[:, None] * X).sum(axis=0) / w.sum()
        if norm(mu_new - mu) <= tol * max(1.0, norm(mu)):
            mu = mu_new
            break
        mu = mu_new
    return mu


def spatial_signs(
    X: np.ndarray,
    center: Literal["spatial_median", "mean"] = "spatial_median",
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute row-wise spatial signs U = (X - center) / ||X - center||.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Data matrix with rows as observations.
    center : {'spatial_median', 'mean'}, default 'spatial_median'
        How to center the data before taking signs.
    eps : float, default 1e-12
        Small constant added to norms to avoid division by zero.
        Rows with norm < eps after centering are dropped.

    Returns
    -------
    U : ndarray of shape (n_kept, n_features)
        Unit-length spatial signs for the kept rows.
    kept_idx : ndarray of shape (n_kept,)
        Indices of rows in X that were kept.
    """
    X = np.asarray(X, dtype=float)
    if center == "spatial_median":
        mu = spatial_median(X)
    elif center == "mean":
        mu = X.mean(axis=0)
    else:
        raise ValueError("center must be 'spatial_median' or 'mean'.")

    Z = X - mu
    r = norm(Z, axis=1)
    kept = r > eps
    if not np.any(kept):
        raise ValueError("All rows have near-zero length after centering.")
    Z = Z[kept]
    r = r[kept]
    U = Z / r[:, None]
    return U, np.nonzero(kept)[0]


def _A_from_gram(G: np.ndarray) -> float:
    """
    Given Gram matrix G = U U^T where U rows are unit vectors,
    compute A = mean_{i != j} (u_i^T u_j)^2.
    """
    n = G.shape[0]
    if n < 2:
        raise ValueError("Need at least 2 rows to compute A.")
    # Frobenius norm squared of G equals sum_{i,j} (u_i^T u_j)^2
    s = np.sum(G * G)
    # Remove diagonal contributions (each diagonal element is 1, squared is 1)
    offdiag_sum = s - n
    return offdiag_sum / (n * (n - 1))


def _C_from_gram_blocks(G: np.ndarray, idx1: np.ndarray, idx2: np.ndarray) -> float:
    """
    Given a pooled Gram matrix G over stacked signs W = [U; V],
    compute C = mean_{i in idx1, j in idx2} (u_i^T v_j)^2.
    """
    G12 = G[np.ix_(idx1, idx2)]
    return np.mean(G12 * G12)


def _T_from_gram(G: np.ndarray, idx1: np.ndarray, idx2: np.ndarray, p: int) -> float:
    """
    Compute T' = p * (A + B - 2*C) from pooled Gram matrix.
    """
    A = _A_from_gram(G[np.ix_(idx1, idx1)])
    B = _A_from_gram(G[np.ix_(idx2, idx2)])
    C = _C_from_gram_blocks(G, idx1, idx2)
    return p * (A + B - 2.0 * C)


def _var_spherical_asymp(n1: int, n2: int, p: int) -> float:
    """
    Asymptotic variance of T' under spherical directions assumption.

    Derivation sketch:
    Let Z = u^T v for independent uniform random unit vectors in R^p.
    E[Z^2] = 1/p and Var(Z^2) = 3/(p(p+2)) - 1/p^2 = 2*(p-1)/(p^2*(p+2)).
    Approximating pair terms as weakly dependent,
    Var(A) ~ Var(Z^2) / [n1 (n1 - 1)], Var(B) similar, Var(C) ~ Var(Z^2) / (n1 n2).
    For D = A + B - 2C, Var(D) ~ Var(A) + Var(B) + 4 Var(C).
    Var(T') = p^2 Var(D) which simplifies to:
        2*(p - 1)/(p + 2) * [ 1/(n1(n1-1)) + 1/(n2(n2-1)) + 4/(n1 n2) ].
    """
    if n1 < 2 or n2 < 2:
        raise ValueError("n1 and n2 must be at least 2 for asymptotics.")
    return (
        2.0
        * (p - 1.0)
        / (p + 2.0)
        * (1.0 / (n1 * (n1 - 1.0)) + 1.0 / (n2 * (n2 - 1.0)) + 4.0 / (n1 * n2))
    )


def _var_empirical_from_gram(
    G: np.ndarray, idx1: np.ndarray, idx2: np.ndarray, p: int
) -> float:
    """
    Empirical plug-in variance of T' using sample variances of pairwise squared cosines.

    This ignores dependence among pairs and treats them as approximately independent
    in large samples, which is a common approximation for U-statistics with pairwise kernels.
    """
    # A block
    G11 = G[np.ix_(idx1, idx1)]
    n1 = len(idx1)
    mask_off_11 = ~np.eye(n1, dtype=bool)
    a_vals = (G11 * G11)[mask_off_11]
    varA = a_vals.var(ddof=1) / (n1 * (n1 - 1))

    # B block
    G22 = G[np.ix_(idx2, idx2)]
    n2 = len(idx2)
    mask_off_22 = ~np.eye(n2, dtype=bool)
    b_vals = (G22 * G22)[mask_off_22]
    varB = b_vals.var(ddof=1) / (n2 * (n2 - 1))

    # C cross block
    G12 = G[np.ix_(idx1, idx2)]
    c_vals = (G12 * G12).ravel()
    varC = c_vals.var(ddof=1) / (n1 * n2)

    varD = varA + varB + 4.0 * varC
    return (p ** 2) * varD


def _perm_pvalue_from_gram(
    G: np.ndarray,
    n1: int,
    n2: int,
    p: int,
    n_perm: int,
    rng: np.random.Generator,
    t_obs: float,
) -> Tuple[float, float]:
    """
    Two-sided and one-sided permutation p-values using precomputed Gram matrix.

    Returns
    -------
    p_two_sided, p_right
    """
    n = n1 + n2
    idx = np.arange(n)
    hits_two = 0
    hits_right = 0
    abs_t_obs = abs(t_obs)

    for _ in range(n_perm):
        rng.shuffle(idx)
        i1 = idx[:n1]
        i2 = idx[n1:]
        t = _T_from_gram(G, i1, i2, p)
        if abs(t) >= abs_t_obs:
            hits_two += 1
        if t >= t_obs:
            hits_right += 1

    # Add one to numerator and denominator for an unbiased finite-sample estimate
    denom = n_perm + 1
    p_two = (hits_two + 1) / denom
    p_right = (hits_right + 1) / denom
    return p_two, p_right
