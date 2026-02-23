import numpy as np
from scipy.stats import norm


def _validate_matrix(X):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array of shape (n_samples, n_features).")
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

    Xc = X - X.mean(axis=0, keepdims=True)

    # Unbiased sample covariance: (1/(n-1)) Xc^T Xc
    S = (Xc.T @ Xc) / (n - 1)

    trS = float(np.trace(S))
    trS2 = float(np.trace(S @ S))  # ||S||_F^2

    # Qi = sum_k (||x_k - xbar||^2)^2
    row_sqnorm = np.sum(Xc * Xc, axis=1)
    Q = float(np.sum(row_sqnorm * row_sqnorm))

    nu = (n - 1) / (n * (n - 2) * (n - 3))

    # Eq. (10): Ei = nu * [ (n-1)(n-2)||S||_F^2 + (tr S)^2 - n*Q ]
    Ei = nu * ((n - 1) * (n - 2) * trS2 + (trS * trS) - n * Q)
    return Ei


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
        raise ValueError("X1 and X2 must have the same number of features (columns).")
    if n1 < 2 or n2 < 2:
        raise ValueError("Need n1 >= 2 and n2 >= 2 to form sample covariances.")

    X1c = X1 - X1.mean(axis=0, keepdims=True)
    X2c = X2 - X2.mean(axis=0, keepdims=True)

    S1 = (X1c.T @ X1c) / (n1 - 1)
    S2 = (X2c.T @ X2c) / (n2 - 1)

    return float(np.trace(S1 @ S2))


def ahmad_2022_proportionality_test(X1, X2, alternative="two-sided"):
    """
    Ahmad (2022) proportionality test H0: Sigma1 = delta Sigma2.

    Test statistic:
        T_hat = E12 / sqrt(E1 * E2) - 1
    Null limit:
        sqrt(n1*n2) * T_hat -> N(0,1) under H0.

    Parameters
    ----------
    X1 : ndarray, shape (n1_samples, n_features)
    X2 : ndarray, shape (n2_samples, n_features)
    alternative : {"two-sided", "greater", "less"}
        - "two-sided": p = 2 * (1 - Phi(|z|))
        - "greater":   p = 1 - Phi(z)
        - "less":      p = Phi(z)

    Returns
    -------
    result : dict with keys
        - "T_hat"
        - "z"
        - "p_value"
        - "E1", "E2", "E12"
    """
    X1 = _validate_matrix(X1)
    X2 = _validate_matrix(X2)
    n1, p1 = X1.shape
    n2, p2 = X2.shape
    if p1 != p2:
        raise ValueError("X1 and X2 must have the same number of features (columns).")

    E1 = estimate_Ei_trSigma2(X1)
    E2 = estimate_Ei_trSigma2(X2)
    E12 = estimate_E12_trSigma1Sigma2(X1, X2)

    if E1 <= 0 or E2 <= 0:
        raise ValueError(
            f"Non-positive E1 or E2 encountered (E1={E1}, E2={E2}). "
            "This can happen with very small n or numerical issues."
        )

    T_hat = (E12 / np.sqrt(E1 * E2)) - 1.0
    z = np.sqrt(n1 * n2) * T_hat

    if alternative == "two-sided":
        p_value = 2.0 * (1.0 - norm.cdf(abs(z)))
    elif alternative == "greater":
        p_value = 1.0 - norm.cdf(z)
    elif alternative == "less":
        p_value = norm.cdf(z)
    else:
        raise ValueError("alternative must be one of: 'two-sided', 'greater', 'less'.")

    return {"T_hat": float(T_hat), "z": float(z), "p_value": float(p_value),
            "E1": float(E1), "E2": float(E2), "E12": float(E12)}

