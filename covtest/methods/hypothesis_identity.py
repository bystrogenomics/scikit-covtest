"""
hypothesis_identity.py
======================

Implementation of hypothesis tests for the identity covariance matrix.

This module provides functions for two classical tests of whether
the covariance matrix of a multivariate distribution equals the
identity matrix:

1. Ledoit–Wolf test (2002).
2. Nagao’s test (1973).

Both tests use statistics based on traces of the sample covariance
matrix. They are useful in high-dimensional hypothesis testing,
multivariate analysis, and as diagnostics for covariance estimation.

References
----------
Ledoit, O. and Wolf, M. (2002).
"Some hypothesis tests for the covariance matrix when the dimension is large
compared to the sample size."
Annals of Statistics, 30(4), 1081–1102.
https://doi.org/10.1214/aos/1031689018

Nagao, H. (1973).
"On some test criteria for covariance matrix."
Annals of Statistics, 1(4), 700–709.
https://doi.org/10.1214/aos/1176342506
"""

import numpy as np
import numpy.linalg as la
import scipy.stats as stats  # type: ignore
from numpy.linalg import slogdet
from scipy.stats import norm

from . import _srivastava_2005 as s2005
from . import _tylers as tyler
from . import _ahmad2015 as ahmad2015
from .utils import (
    covariance_traces,
    result_dict,
    sample_covariance,
    validate_data_matrix,
)


def test_identity_T2(
    X: np.ndarray,
    center: bool = True,
    calibration: str = "auto",
    tail: str = "upper",
):
    """
    Test H0: Sigma = I (identity) using T2 = (1/p)*E3 - (2/p)*E1 + 1.

    Parameters are analogous to test_sphericity_T1.

    Note: For a general null Sigma = Sigma0, you typically whiten first:
          X_whitened = X @ Sigma0^{-1/2}  (or apply a linear transform with that effect)
          then run test_identity_T2 on X_whitened.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array of shape (n, p).")
    n, p = X.shape
    if n < 2:
        raise ValueError("Need n >= 2 samples.")
    if p < 1:
        raise ValueError("Need p >= 1 variables.")

    if center:
        X = X - X.mean(axis=0, keepdims=True)

    G = X @ X.T
    E1, E2, E3 = ahmad2015._trace_estimators_from_gram(G)

    T2 = (E3 / p) - (2.0 * E1 / p) + 1.0

    z, used_cal = ahmad2015._standardize_T(
        T2, n=n, p=p, calibration=calibration
    )

    if tail == "upper":
        pval = float(norm.sf(z))
    elif tail == "two-sided":
        pval = float(2.0 * norm.sf(abs(z)))
    else:
        raise ValueError("tail must be 'upper' or 'two-sided'.")

    return {
        "stat": float(T2),
        "p_value": pval,
    }


def _ledoit_wolf_stat(data):
    """Compute the Ledoit–Wolf test statistic.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_features)
        The data matrix, where rows correspond to samples and
        columns to variables.

    Returns
    -------
    W : float
        The Ledoit–Wolf test statistic.

    Notes
    -----
    Let :math:`S` denote the sample covariance matrix and
    :math:`p` the number of features.
    Define

    .. math::

        W = \\frac{1}{p} \\operatorname{tr}[(S - I_p)^2]
            - \\frac{1}{np} \\big( \\operatorname{tr}(S) \\big)^2
            + \\frac{p}{n},

    where :math:`I_p` is the :math:`p \\times p` identity matrix, and
    :math:`n` is the number of samples.

    The statistic :math:`W` is used to form the Ledoit–Wolf test
    for :math:`H_0 : \\Sigma = I_p`.
    """
    n, p = data.shape
    sample_cov_matrix, trace_S, _ = covariance_traces(data)
    SmI = sample_cov_matrix - np.eye(p)
    trace_smi2 = np.trace(SmI @ SmI)
    W = 1 / p * trace_smi2 - 1 / (n * p) * trace_S**2 + p / n
    return W


# Checked
def ledoit_wolf_identity(X):
    """Ledoit-Wolf test for identity covariance matrix.

    Tests the null hypothesis H0: Sigma = I_p using the
    Ledoit-Wolf statistic.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data matrix where rows are samples and columns are features.

    Returns
    -------
    result : dict
        Dictionary with keys:

        - 'stat' : float
            Chi-square test statistic.
        - 'p_value' : float
            P-value from chi-square distribution.

    References
    ----------
    .. [1] Ledoit, O., & Wolf, M. (2002). "Some hypothesis tests
           for the covariance matrix when the dimension is large
           compared to the sample size." Annals of Statistics,
           30(4), 1081-1102.
    """
    X = validate_data_matrix(X)
    n, p = X.shape
    W = _ledoit_wolf_stat(X)
    degree_of_freedom = p * (p + 1) / 2
    stat = n * p / 2 * W
    p_value = 1 - stats.chi2.cdf(stat, degree_of_freedom)
    return result_dict(stat, p_value)


def _nagao_stat(data):
    """Compute Nagao’s test statistic.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_features)
        The data matrix, where rows correspond to samples and
        columns to variables.

    Returns
    -------
    V : float
        Nagao’s test statistic.

    Notes
    -----
    Let :math:`S` denote the sample covariance matrix and :math:`p` the number of features.
    Define

    .. math::

        V = \\frac{1}{p} \\operatorname{tr}(S^2)
            - \\frac{2}{p} \\operatorname{tr}(S) + 1.

    The statistic :math:`V` is used to form Nagao’s test
    for :math:`H_0 : \\Sigma = I_p`.
    """
    _, trace_S, trace_S2 = covariance_traces(data)
    p = data.shape[1]
    V = 1 / p * trace_S2 - 2 / p * trace_S + 1
    return V


# Checked
def nagao_identity(X):
    """Nagao's test for identity covariance matrix.

    Tests the null hypothesis H0: Sigma = I_p.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data matrix where rows are samples and columns are features.

    Returns
    -------
    result : dict
        Dictionary with keys:

        - 'stat' : float
            Chi-square test statistic.
        - 'p_value' : float
            P-value from chi-square distribution.

    Notes
    -----
    The test statistic is T = (np/2) * V, where V is Nagao's
    statistic based on traces of the sample covariance matrix.

    References
    ----------
    .. [1] Nagao, H. (1973). "On some test criteria for covariance
           matrix." Annals of Statistics, 1(4), 700-709.
    """
    X = validate_data_matrix(X)
    n, p = X.shape
    V = _nagao_stat(X)
    degree_of_freedom = p * (p + 1) / 2
    stat = n * p / 2 * V
    p_value = 1 - stats.chi2.cdf(stat, degree_of_freedom)
    return result_dict(stat, p_value)


# Checked
def srivastava_2005_identity(X):
    """Srivastava (2005) test for identity covariance matrix.

    High-dimensional test for H0: Sigma = I_p.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data matrix where rows are samples and columns are features.

    Returns
    -------
    result : dict
        Dictionary with keys:

        - 'stat' : float
            Test statistic (asymptotically normal).
        - 'p_value' : float
            Right-tail p-value from standard normal distribution.

    References
    ----------
    .. [1] Srivastava, M. S. (2005). "Some tests concerning the
           covariance matrix in high dimensional data." Journal of
           the Japan Statistical Society, 35(2), 251-272.
    """
    X = validate_data_matrix(X)
    n = X.shape[0]
    S = sample_covariance(X)
    T_1 = s2005.T_1_stat(S, n)
    z_stat = (n / 2) * T_1
    p_value = 1 - stats.norm.cdf(z_stat)
    return result_dict(z_stat, p_value)


def tyler_identity(X, unknown_mean=False, method="tr"):
    """Tyler's M-estimator test for identity shape matrix.

    Tests H0: Sigma = I_p using Tyler's shape matrix estimator.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data matrix where rows are samples and columns are features.
    unknown_mean : bool, default=False
        If True, uses robust location estimation before testing.
    method : {'tr', 'log'}, default='tr'
        Test statistic type: 'tr' for trace-based, 'log' for
        log-determinant-based.

    Returns
    -------
    result : dict
        Dictionary with keys:

        - 'stat' : float
            Standardized test statistic.
        - 'p_value' : float
            Right-tail p-value from standard normal distribution.

    References
    ----------
    .. [1] Tyler, D. E. (1987). "A distribution-free M-estimator
           of multivariate scatter." Annals of Statistics, 15(1),
           234-251.
    """
    X = validate_data_matrix(X)
    n, p = X.shape
    if unknown_mean:
        mu_hat = tyler.robust_location(X)
        Xc = X - mu_hat
        C = tyler.tylers_M(Xc)
        c = p / (n - 1)
        T_tr = np.trace(C @ C)
        T_log = slogdet(C)[1]

        # Null means/variances (Theorem 2.6)
        mean_tr = p * (1 + p / (n - 1))
        var_tr = 4 * c**2
        mean_log = -(p - (n - 1)) * np.log(1 - c) - p
        var_log = -2 * np.log(1 - c) - 2 * c

    else:
        C = tyler.tylers_M(X)
        c = p / n
        T_tr = np.trace(C @ C)
        T_log = slogdet(C)[1]

        # Null means/variances (Theorem 2.1)
        mean_tr = p * (1 + c) - (c * (c - 3)) / (1 - c)
        var_tr = (2 * c) ** 2
        mean_log = (
            -p + (p - n) * np.log(1 - c) + 0.5 * np.log(1 - c) - c / (1 - c)
        )
        var_log = -2 * np.log(1 - c) - 2 * c

    z_tr = (T_tr - mean_tr) / np.sqrt(var_tr)
    z_log = (T_log - mean_log) / np.sqrt(var_log)
    if method == "tr":
        return result_dict(z_tr, 1 - stats.norm.cdf(z_tr))
    return result_dict(z_log, 1 - stats.norm.cdf(z_log))


def _covariance_under_null(S, Sigma):
    if isinstance(Sigma, str) and Sigma == "identity":
        return S

    sv = la.svd(Sigma)
    svDf = la.svd(S)
    sqrt_sv = np.diag(np.sqrt(sv[1]))
    x_ = svDf[0] @ sqrt_sv @ la.inv(sv[0] @ sqrt_sv)
    return x_.T @ x_


def _fisher_2012_stat_(n, p, S_):
    c = p / n
    ahat2 = (n**2 / ((n - 1) * (n + 2) * p)) * (
        np.sum(np.diag(S_ @ S_)) - (np.sum(np.diag(S_)) ** 2) / n
    )
    gamma = (n**5 * (n**2 + n + 2)) / (
        (n + 1) * (n + 2) * (n + 4) * (n + 6) * (n - 1) * (n - 2) * (n - 3)
    )
    ahat4 = (gamma / p) * (
        np.sum(np.diag(S_ @ S_ @ S_ @ S_))
        - (4 / n) * np.sum(np.diag(S_ @ S_ @ S_)) * np.sum(np.diag(S_))
        - ((2 * (n**2) + 3 * n - 6) / (n * (n**2 + n + 2)))
        * (np.sum(np.diag(S_ @ S_)) ** 2)
        + ((2 * (5 * n + 6)) / (n * (n**2 + n + 2)))
        * np.sum(np.diag(S_ @ S_))
        * (np.sum(np.diag(S_)) ** 2)
        - ((5 * n + 6) / ((n**2) * (n**2 + n + 2))) * (np.sum(np.diag(S_)) ** 4)
    )
    return (n / np.sqrt(8 * (c**2 + 12 * c + 8))) * (ahat4 - 2 * ahat2 + 1)


def fisher_single_sample(X, Sigma="identity"):
    """Fisher's test for covariance matrix structure.

    Tests the null hypothesis that the covariance matrix equals
    a specified matrix.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data matrix where rows are samples and columns are features.
    Sigma : {"identity", array-like}, default="identity"
        Hypothesized covariance matrix.

    Returns
    -------
    result : dict
        Dictionary with keys:

        - 'stat' : float
            Test statistic.
        - 'p_value' : float
            Two-sided p-value.

    References
    ----------
    .. [1] Fisher, T. J., et al. (2012). "A high-dimensional test
           for the equality of the smallest eigenvalues of a
           covariance matrix." Journal of Multivariate Analysis.
    """
    X = validate_data_matrix(X)
    p = X.shape[1]
    n = X.shape[0]
    S = sample_covariance(X)
    S_ = _covariance_under_null(S, Sigma)

    statistic = _fisher_2012_stat_(n - 1, p, S_)
    p_value = 2 * (1 - norm.cdf(abs(statistic)))

    return result_dict(statistic, p_value)


def _srivastava2011_(n, p, S_):
    term1 = (
        (n**2 / ((n - 1) * (n + 2)))
        * (np.trace(S_ @ S_) - np.trace(S_) ** 2 / n)
        / p
    )
    term2 = 2 * (np.trace(S_) / p)
    return n * (term1 - term2 + 1) / 2


def srivastava2011_single_sample(X, Sigma="identity"):
    """Srivastava (2011) test for covariance matrix structure.

    High-dimensional test for covariance matrix equality.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data matrix where rows are samples and columns are features.
    Sigma : {"identity", array-like}, default="identity"
        Hypothesized covariance matrix.

    Returns
    -------
    result : dict
        Dictionary with keys:

        - 'stat' : float
            Test statistic.
        - 'p_value' : float
            Two-sided p-value.

    """
    X = validate_data_matrix(X)
    p = X.shape[1]
    n = X.shape[0]
    S = sample_covariance(X)
    S_ = _covariance_under_null(S, Sigma)

    statistic = _srivastava2011_(n - 1, p, S_)
    p_value = 2 * (1 - norm.cdf(abs(statistic)))

    return result_dict(statistic, p_value)


def one_sample_cov_test(X, mean=None, S=None):
    """
    Perform a one-sample covariance test to evaluate if the sample
    covariance differs
    from the identity matrix or another specified covariance matrix.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        The input data matrix.
    mean : ndarray of shape (n_features,), default=None
        The mean vector for adjusting the data. If None, use the sample mean.
    S : ndarray of shape (n_features, n_features), default=None
        The covariance matrix for the test. If None, the identity matrix is assumed.

    Returns
    -------
    dict
        A dictionary containing:
        - 'p_value': The p-value of the test.
        - 'z_value': The computed Z-value for the test.
        - 'lrt': The likelihood ratio test statistic.
    """
    n, p = X.shape
    y = p / n
    N = n - 1
    yN = p / N

    if S is not None:
        S_half = la.cholesky(S)
        X = X @ la.inv(S_half)

    if mean is None:
        X = X - np.mean(X, axis=0)
        S = X.T @ X / N
    else:
        X = X - mean
        S = X.T @ X / n

    lrt = np.sum(np.diag(S)) - np.log(la.det(S)) - p
    mu1 = -0.5 * np.log(1 - y)
    sigma1 = -2 * np.log(1 - y) - 2 * y
    z_value = (lrt - p * (1 + (1 - yN) / yN * np.log(1 - yN)) - mu1) / np.sqrt(
        sigma1
    )
    p_value = norm.sf(z_value)

    return {"p_value": p_value, "z_value": z_value, "lrt": lrt}
