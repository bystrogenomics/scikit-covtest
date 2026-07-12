"""
_ahmad.py
=========

Consolidated Ahmad helper functions, combining material from the three
original per-paper modules:

  * _ahmad2015 – U-statistic trace estimators and T-statistic calibration
                 (Ahmad & von Rosen, 2015).
  * _ahmad2017 – location-invariant estimators of tr(Σᵢ²) and tr(Σ₁Σ₂)
                 (Ahmad, 2017).
  * _ahmad2022 – validated wrappers for the 2017 estimators used by the
                 proportionality tests (Ahmad, 2022).

References
----------
Ahmad, M. R. & von Rosen, D. (2015).
    Communications in Statistics – Theory and Methods 44(7), 1387-1398.
Ahmad, M. R. (2017).
    Scandinavian Journal of Statistics 44, 500-523.
Ahmad, M. R. (2022).
    Journal of Multivariate Analysis 188, 104827.
"""

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Shared validation
# ─────────────────────────────────────────────────────────────────────────────


def _validate_matrix(X):
    """Return X as a finite 2-D float array, raising ValueError otherwise."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(
            "X must be a 2D array of shape (n_samples, n_features)."
        )
    if not np.isfinite(X).all():
        raise ValueError("X contains NaN/Inf.")
    return X


# ─────────────────────────────────────────────────────────────────────────────
# Ahmad & von Rosen (2015) – Gram-matrix U-statistic estimators
# ─────────────────────────────────────────────────────────────────────────────


def _trace_estimators_from_gram(G: np.ndarray):
    """
    Given Gram matrix G = X X^T (n × n), compute U-statistic trace estimators.

    E1 = (1/n) Σ_k A_k                      estimates tr(Σ)
    E2 = (1/(n(n-1))) Σ_{k≠l} A_k A_l      estimates (tr(Σ))²
    E3 = (1/(n(n-1))) Σ_{k≠l} (A_kl)²      estimates tr(Σ²)

    where A_k = G_kk and A_kl = G_kl.
    """
    n = G.shape[0]
    if G.shape[1] != n:
        raise ValueError("G must be square (n x n).")
    if n < 2:
        raise ValueError("Need n >= 2 samples.")

    diag = np.diag(G)
    sum_diag = float(np.sum(diag))
    sum_diag2 = float(np.sum(diag * diag))

    E1 = sum_diag / n

    # Σ_{k≠l} A_k A_l = (Σ_k A_k)² − Σ_k A_k²
    E2 = ((sum_diag * sum_diag) - sum_diag2) / (n * (n - 1))

    # Σ_{k≠l} (A_kl)² = Σ_{k,l} G_kl² − Σ_k G_kk²
    G2_sum = float(np.sum(G * G))
    E3 = (G2_sum - sum_diag2) / (n * (n - 1))

    return E1, E2, E3


def _standardize_T(T: float, n: int, p: int, calibration: str = "ahmad2015"):
    """
    Standardize an Ahmad/von Rosen (2015) T statistic under the null.

    For both the sphericity statistic T1 and identity statistic T2 the null
    limit is:

        (n / 2) * T → N(0, 1)

    so the default z-score is z = (n / 2) * T.

    The legacy name ``large_p_small_n`` is retained as an alias for
    ``ahmad2015``.  The ``ratio`` calibration is an explicit opt-in only and
    is not selected by ``auto``.
    """
    allowed = {"auto", "ahmad2015", "large_p_small_n", "ratio"}
    if calibration not in allowed:
        raise ValueError(
            "calibration must be one of: auto, ahmad2015, large_p_small_n, ratio"
        )

    if calibration in {"auto", "ahmad2015", "large_p_small_n"}:
        return (n / 2.0) * T, "ahmad2015"

    c = p / n
    var_nT = 4.0 * (1.0 + 2.0 / c)
    return (n * T) / np.sqrt(var_nT), "ratio"


# ─────────────────────────────────────────────────────────────────────────────
# Ahmad (2017) – location-invariant estimators for two-sample tests
# ─────────────────────────────────────────────────────────────────────────────


def estimate_Ei(X):
    """
    Location-invariant estimator of tr(Σᵢ²) from sample X.

    Implements the formula from Ahmad (2017) that is unbiased under the
    high-dimensional asymptotic regime.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Raw (un-centred) data from population i.

    Returns
    -------
    Ei : float
        Estimate of tr(Σᵢ²).
    """
    X = _validate_matrix(X)
    n, p = X.shape
    if n < 4:
        raise ValueError(
            "Need n >= 4 for estimate_Ei (denominator (n-2)(n-3))."
        )
    X_centered = X - X.mean(axis=0)
    S = np.cov(X_centered, rowvar=False, bias=False)
    trace_S = np.trace(S)
    trace_S2 = np.trace(S @ S)
    Q = np.sum(np.sum(X_centered**2, axis=1) ** 2) / (n - 1)
    eta = (n - 1) / (n * (n - 2) * (n - 3))
    return eta * ((n - 1) * (n - 2) * trace_S2 + trace_S**2 - n * Q)


def estimate_E12(X, Y):
    """
    Estimator of tr(Σ₁ Σ₂) using cross-covariance.

    Parameters
    ----------
    X : array-like of shape (n1_samples, n_features)
    Y : array-like of shape (n2_samples, n_features)

    Returns
    -------
    E12 : float
        Estimate of tr(Σ₁ Σ₂).
    """
    X = _validate_matrix(X)
    Y = _validate_matrix(Y)
    n1, p1 = X.shape
    n2, p2 = Y.shape
    if p1 != p2:
        raise ValueError(
            "X and Y must have the same number of features (columns)."
        )
    if n1 < 2 or n2 < 2:
        raise ValueError("Need n1 >= 2 and n2 >= 2 to form sample covariances.")
    X_centered = X - X.mean(axis=0)
    Y_centered = Y - Y.mean(axis=0)
    S1 = np.cov(X_centered, rowvar=False, bias=False)
    S2 = np.cov(Y_centered, rowvar=False, bias=False)
    return float(np.trace(S1 @ S2))


# ─────────────────────────────────────────────────────────────────────────────
# Ahmad (2022) – validated estimators used by proportionality tests
# ─────────────────────────────────────────────────────────────────────────────


def estimate_Ei_trSigma2(X):
    """
    Ahmad (2022) estimator Eᵢ of tr(Σᵢ²) = ‖Σᵢ‖_F².

    Delegates to :func:`estimate_Ei` after validating the input.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)

    Returns
    -------
    Ei : float
        Unbiased/consistent estimator (Eq. 10 of Ahmad 2022).
    """
    X = _validate_matrix(X)
    n, p = X.shape
    if n < 4:
        raise ValueError("Need n >= 4 for Ei because of (n-2)(n-3) in nu_i.")
    return estimate_Ei(X)


def estimate_E12_trSigma1Sigma2(X1, X2):
    """
    Ahmad (2022) estimator E₁₂ of tr(Σ₁ Σ₂) (Eq. 9).

    Delegates to :func:`estimate_E12` after validating both inputs.

    Parameters
    ----------
    X1 : array-like of shape (n1_samples, n_features)
    X2 : array-like of shape (n2_samples, n_features)

    Returns
    -------
    E12 : float
    """
    # validate_matrix and dimension checks happen inside estimate_E12
    return estimate_E12(X1, X2)
