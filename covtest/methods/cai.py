import numpy as np


def _validate_matrix(X, name):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"{name} must be a 2D array (n_samples, n_features).")
    if not np.isfinite(X).all():
        raise ValueError(f"{name} contains NaN/Inf.")
    return X


def _center(X):
    return X - X.mean(axis=0, keepdims=True)


def _sample_cov_unbiased(Xc):
    # Unbiased sample covariance with 1/n (not 1/(n-1))? The paper defines 1/n in Sec. 2.
    # We'll follow the paper literally: \hat Sigma = (1/n) sum (x_k - xbar)(x_k - xbar)^T.
    # (This differs from the usual unbiased estimator, but matches their definition.)
    n = Xc.shape[0]
    return (Xc.T @ Xc) / n


def _theta_hat_matrix(Xc, S_hat):
    """
    Compute theta_hat_ij = (1/n) sum_k [ (Xki Xkj - S_hat_ij) ]^2
    using centered data Xc and S_hat = (1/n) Xc^T Xc.

    This matches the paper's variance estimator in Section 2. :contentReference[oaicite:7]{index=7}
    """
    n, p = Xc.shape
    # For each k, we need outer product of row k: x_k x_k^T (p x p),
    # then subtract S_hat and square elementwise, then average over k.
    theta = np.zeros((p, p), dtype=float)
    for k in range(n):
        outer = np.outer(Xc[k], Xc[k])
        diff = outer - S_hat
        theta += diff * diff
    theta /= n
    return theta


def cai_liu_xia_2013_two_sample_test(X, Y, alpha=0.05, use_upper_triangle=True, eps=1e-12):
    """
    Cai, Liu & Xia (2013) two-sample covariance test (max-type).

    Parameters
    ----------
    X : ndarray, shape (n1, p)
    Y : ndarray, shape (n2, p)
    alpha : float
        Significance level used for the critical value (Eq. (4)-(5)). :contentReference[oaicite:8]{index=8}
    use_upper_triangle : bool
        If True, compute max over i<=j only (as in the paper). :contentReference[oaicite:9]{index=9}
        If False, compute max over all i,j (gives the same max for symmetric matrices).
    eps : float
        Small jitter to avoid division by zero in pathological cases.

    Returns
    -------
    result : dict with keys:
        - Mn : float, max statistic (Eq. (3))
        - t  : float, centered/scaled value Mn - 4 log p + log log p
        - p_value : float, asymptotic p-value from Theorem 1 / Eq. (9)
        - critical_value : float, q_alpha + 4 log p - log log p (Eq. (4)-(5))
        - reject : bool, Mn >= critical_value
    """
    X = _validate_matrix(X, "X")
    Y = _validate_matrix(Y, "Y")
    n1, p = X.shape
    n2, p2 = Y.shape
    if p2 != p:
        raise ValueError("X and Y must have the same number of features (columns).")
    if n1 < 2 or n2 < 2:
        raise ValueError("Need n1 >= 2 and n2 >= 2.")

    # Center
    Xc = _center(X)
    Yc = _center(Y)

    # Paper's sample covariances: (1/n) sum (x - xbar)(x - xbar)^T. :contentReference[oaicite:10]{index=10}
    S1 = _sample_cov_unbiased(Xc)
    S2 = _sample_cov_unbiased(Yc)

    # theta-hat matrices (Section 2). :contentReference[oaicite:11]{index=11}
    theta1 = _theta_hat_matrix(Xc, S1)
    theta2 = _theta_hat_matrix(Yc, S2)

    # Denominator: theta1/n1 + theta2/n2
    denom = theta1 / n1 + theta2 / n2
    denom = np.maximum(denom, eps)

    diff = S1 - S2
    M = (diff * diff) / denom  # M_ij (Eq. (2)) :contentReference[oaicite:12]{index=12}

    if use_upper_triangle:
        iu = np.triu_indices(p)
        Mn = float(np.max(M[iu]))
    else:
        Mn = float(np.max(M))

    # Asymptotic null scaling (Theorem 1 / Eq. (9)) :contentReference[oaicite:13]{index=13}
    t = Mn - 4.0 * np.log(p) + np.log(np.log(p))

    # Limiting CDF F(t) = exp( - (1/sqrt(8pi)) exp(-t/2) ) (Eq. (9)) :contentReference[oaicite:14]{index=14}
    F_t = np.exp(-(1.0 / np.sqrt(8.0 * np.pi)) * np.exp(-t / 2.0))
    p_value = float(1.0 - F_t)  # upper tail

    # Critical value via q_alpha (Eq. (4)-(5)) :contentReference[oaicite:15]{index=15}
    # q_alpha = -log(8pi) - 2 log( log((1-alpha)^{-1}) )
    q_alpha = -np.log(8.0 * np.pi) - 2.0 * np.log(np.log(1.0 / (1.0 - alpha)))
    critical_value = float(q_alpha + 4.0 * np.log(p) - np.log(np.log(p)))

    reject = Mn >= critical_value

    return {
        "Mn": float(Mn),
        "t": float(t),
        "p_value": p_value,
        "critical_value": critical_value,
        "reject": bool(reject),
        "n1": int(n1),
        "n2": int(n2),
        "p": int(p),
    }

