import numpy as np
from numpy.linalg import norm


def _center_and_scale(X):
    """Center the data and return unit vectors and norms."""
    mu = np.median(X, axis=0)
    Z = X - mu
    norms = norm(Z, axis=1, keepdims=True)
    U = Z / norms
    return U, norms.squeeze()


def _compute_statistic(U, scores, k, score_var):
    G = U @ U.T
    S = G**2 - 1 / k
    Q = scores @ S @ scores
    return (k * (k + 2)) / (2 * U.shape[0] * score_var) * Q
