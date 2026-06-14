import numpy as np
from . import _ahmad2017 as ahmad2017


def _validate_matrix(X):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(
            "X must be a 2D array of shape (n_samples, n_features)."
        )
    if not np.isfinite(X).all():
        raise ValueError("X contains NaN/Inf.")
    return X


def estimate_Ei_trSigma2(X):
    """
    Ahmad (2022) estimator Ei of tr(Sigma_i^2) = ||Sigma_i||_F^2.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)

    Returns
    -------
    Ei : float
        Unbiased/consistent estimator in Eq. (10).
    """
    X = _validate_matrix(X)
    n, p = X.shape
    if n < 4:
        raise ValueError("Need n >= 4 for Ei because of (n-2)(n-3) in nu_i.")

    return ahmad2017.estimate_Ei(X)


def estimate_E12_trSigma1Sigma2(X1, X2):
    """
    Ahmad (2022) estimator E12 of tr(Sigma1 Sigma2) in Eq. (9),
    using tr(S1 S2) with unbiased sample covariances.

    Parameters
    ----------
    X1 : ndarray, shape (n1_samples, n_features)
    X2 : ndarray, shape (n2_samples, n_features)

    Returns
    -------
    E12 : float
    """
    X1 = _validate_matrix(X1)
    X2 = _validate_matrix(X2)
    n1, p1 = X1.shape
    n2, p2 = X2.shape
    if p1 != p2:
        raise ValueError(
            "X1 and X2 must have the same number of features (columns)."
        )
    if n1 < 2 or n2 < 2:
        raise ValueError("Need n1 >= 2 and n2 >= 2 to form sample covariances.")

    return ahmad2017.estimate_E12(X1, X2)
