import numpy as np


def _validate_matrix(X, name):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"{name} must be 2D with shape (n_samples, n_features).")
    if not np.isfinite(X).all():
        raise ValueError(f"{name} contains NaN/Inf.")
    return X


def _center(X):
    return X - X.mean(axis=0, keepdims=True)


def _cov_and_theta_hat(Xc):
    """
    Chang et al. use:
      - sample covariance:  S = (1/n) sum_i x_i x_i^T  (with x_i centered)
      - variance estimator: theta_{jk} = (1/n) sum_i (x_{ij}x_{ik} - S_{jk})^2

    We compute theta efficiently via:
      E[(x_j x_k)^2] - (E[x_j x_k])^2
    where E[(x_j x_k)^2] = (1/n) sum_i x_{ij}^2 x_{ik}^2.

    Returns
    -------
    S : (p,p)
    theta : (p,p)
    """
    n, p = Xc.shape

    # S_{jk} = (1/n) sum_i x_{ij} x_{ik}
    S = (Xc.T @ Xc) / n

    # E[(x_j x_k)^2] = (1/n) sum_i x_{ij}^2 x_{ik}^2
    Xsq = Xc * Xc
    E_sq = (Xsq.T @ Xsq) / n

    theta = E_sq - (S * S)
    # Numerical guard (finite-sample can yield tiny negative due to roundoff)
    theta = np.maximum(theta, 0.0)
    return S, theta
