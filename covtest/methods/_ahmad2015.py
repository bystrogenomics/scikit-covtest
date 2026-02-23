import numpy as np
from scipy.stats import norm
from scipy.linalg import cholesky, solve_triangular


def _center_columns(X: np.ndarray) -> np.ndarray:
    return X - X.mean(axis=0, keepdims=True)


def _whiten_by_sigma0(X: np.ndarray, sigma0: np.ndarray) -> np.ndarray:
    """
    Apply Sigma0^{-1/2} on the right: X -> X * Sigma0^{-1/2}.
    Uses Cholesky Sigma0 = L L^T and triangular solves.
    """
    L = cholesky(sigma0, lower=True, check_finite=False)
    Xt = X.T
    Zt = solve_triangular(L, Xt, lower=True, check_finite=False)  # L * Zt = Xt
    return Zt.T
