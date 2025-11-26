import numpy as np


def estimate_Ei(X):
    """
    Compute estimator of tr(Sigma_i^2) from sample X using location-invariant formula.
    """
    n, p = X.shape
    X_centered = X - X.mean(axis=0)
    S = np.cov(X_centered, rowvar=False, bias=False)
    trace_S = np.trace(S)
    trace_S2 = np.trace(S @ S)
    Q = np.sum(np.sum(X_centered**2, axis=1) ** 2) / (n - 1)
    eta = (n - 1) / (n * (n - 2) * (n - 3))
    Ei = eta * ((n - 1) * (n - 2) * trace_S2 + trace_S**2 - n * Q)
    return Ei


def estimate_E12(X, Y):
    """
    Compute estimator of tr(Sigma1 * Sigma2) using cross-covariance.
    """
    X_centered = X - X.mean(axis=0)
    Y_centered = Y - Y.mean(axis=0)
    S1 = np.cov(X_centered, rowvar=False, bias=False)
    S2 = np.cov(Y_centered, rowvar=False, bias=False)
    return np.trace(S1 @ S2)
