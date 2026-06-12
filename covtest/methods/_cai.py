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
