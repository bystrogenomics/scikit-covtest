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
from numpy.linalg import slogdet, solve, svd
from scipy.stats import norm

from . import _srivastava_2005 as s2005
from . import _tylers as tyler


def _ahmad_2015_stat(x: np.ndarray) -> float:
    """
    x : (n, p) array, rows = observations, cols = variables.
        If testing against non-identity Sigma, whiten x before calling.
    """
    x = np.asarray(x, dtype=float)
    nrow, ncol = x.shape

    G = x @ x.T
    diagG = np.diag(G)

    c1 = diagG.mean()

    total_sq = np.sum(G**2)
    diag_sq = np.sum(diagG**2)
    off_diag_sq = total_sq - diag_sq

    c3 = off_diag_sq / (nrow * (nrow - 1))

    return nrow * (c3 / ncol - 2.0 * c1 / ncol + 1.0)


def ahmad2015_identity(x, Sigma="identity"):
    """
    Ahmad & von Rosen (2015) test of covariance matrix structure,
    when a data matrix x (n x p) is supplied.
    """
    n, p = x.shape

    if Sigma == "identity":
        x_ = x
    else:
        u_s, d_s, _ = svd(Sigma)
        x_ = x @ solve(u_s @ np.diag(np.sqrt(d_s)), np.eye(p))

    statistic = _ahmad_2015_stat(x_)
    parameter = {"Mean": 0, "Variance": 4 * (2 / (p / n + 1))}
    pval = 2 * (
        1
        - stats.norm.cdf(
            np.abs(statistic), loc=0, scale=np.sqrt(parameter["Variance"])
        )
    )

    return {"stat": statistic, "p_value": pval}


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
    sample_cov_matrix = np.cov(data, rowvar=False)
    trace_S = np.trace(sample_cov_matrix)
    SmI = sample_cov_matrix - np.eye(p)
    trace_smi2 = np.trace(np.dot(SmI, SmI))
    trace_S = np.trace(sample_cov_matrix)
    W = 1 / p * trace_smi2 - 1 / (n * p) * trace_S**2 + p / n
    return W


# Checked
def ledoit_wolf_identity(data):
    """Perform the Ledoit–Wolf test for identity covariance.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_features)
        The data matrix, where rows correspond to samples and
        columns to variables.

    Returns
    -------
    results : dict
        Dictionary containing the following keys:

        - ``'stat'`` : float
            The value of the test statistic.
        - ``'p_value'`` : float
            The p-value from the chi-square distribution.
    """
    n, p = data.shape
    W = _ledoit_wolf_stat(data)
    degree_of_freedom = p * (p + 1) / 2
    stat = n * p / 2 * W
    p_value = 1 - stats.chi2.cdf(stat, degree_of_freedom)
    results = {"stat": stat, "p_value": p_value}
    return results


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
    n, p = data.shape
    sample_cov_matrix = np.cov(data, rowvar=False)
    trace_S = np.trace(sample_cov_matrix)
    trace_S2 = np.trace(np.dot(sample_cov_matrix, sample_cov_matrix))
    V = 1 / p * trace_S2 - 2 / p * trace_S + 1
    return V


# Checked
def nagao_identity(data):
    """Perform Nagao’s test for identity covariance.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_features)
        The data matrix, where rows correspond to samples and columns to variables.

    Returns
    -------
    results : dict
        Dictionary containing the following keys:

        - ``'stat'`` : float
            The value of the test statistic.
        - ``'p_value'`` : float
            The p-value from the chi-square distribution.

    Notes
    -----
    The test statistic is defined as:

    .. math::

        T = \\frac{np}{2} V

    where :math:`V` is Nagao’s test statistic.

    The null hypothesis is :math:`\\Sigma = I_p`, where :math:`\\Sigma` is the covariance
    matrix and :math:`I_p` is the identity matrix.
    """
    n, p = data.shape
    V = _nagao_stat(data)
    degree_of_freedom = p * (p + 1) / 2
    stat = n * p / 2 * V
    p_value = 1 - stats.chi2.cdf(stat, degree_of_freedom)
    results = {"stat": stat, "p_value": p_value}
    return results


# Checked
def srivastava_2005_identity(X):
    n = X.shape[0]
    S = np.cov(X.T)
    T_1 = s2005.T_1_stat(S, n)
    z_stat = (n / 2) * T_1
    p_value = 1 - stats.norm.cdf(z_stat)
    results = {"stat": z_stat, "p_value": p_value}
    return results


def tyler_identity(X, unknown_mean=False, method="tr"):
    """
    Tests for Large-Dimensional Shape Matrices via Tyler’s M Estimators

    One-sample test H0: Sigma = I_p.
    If unknown_mean=True, uses robust location-adjusted version.
    """
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
        return {"stat": z_tr, "p_value": 1 - stats.norm.cdf(z_tr)}
    else:
        return {"stat": z_log, "p_value": 1 - stats.norm.cdf(z_log)}


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
        - ((5 * n + 6) / ((n**2) * (n**2 + n + 2)))
        * (np.sum(np.diag(S_)) ** 4)
    )
    return (n / np.sqrt(8 * (c**2 + 12 * c + 8))) * (ahat4 - 2 * ahat2 + 1)


def fisher_single_sample(x, Sigma="identity"):
    p = x.shape[1]
    n = x.shape[0]
    S = np.cov(x, rowvar=False)

    if Sigma == "identity":
        S_ = S
    else:
        sv = la.svd(Sigma)
        svDf = la.svd(S)
        x_ = (
            svDf[0]
            @ np.diag(np.sqrt(sv[1]))
            @ la.inv(sv[0] @ np.diag(np.sqrt(sv[1])))
        )
        S_ = x_.T @ x_

    statistic = _fisher_2012_stat_(n - 1, p, S_)
    p_value = 2 * (1 - norm.cdf(abs(statistic)))

    return {
        "stat": statistic,
        "p_value": p_value,
    }


def _srivastava2011_(n, p, S_):
    term1 = (
        (n**2 / ((n - 1) * (n + 2)))
        * (np.trace(S_ @ S_) - np.trace(S_) ** 2 / n)
        / p
    )
    term2 = 2 * (np.trace(S_) / p)
    return n * (term1 - term2 + 1) / 2


def srivastava2011_single_sample(x, Sigma="identity"):
    p = x.shape[1]
    n = x.shape[0]
    S = np.cov(x, rowvar=False)

    if Sigma == "identity":
        S_ = S
    else:
        sv = np.linalg.svd(Sigma)
        svDf = np.linalg.svd(S)
        x_ = (
            svDf[0]
            @ np.diag(np.sqrt(sv[1]))
            @ np.linalg.inv(sv[0] @ np.diag(np.sqrt(sv[1])))
        )
        S_ = x_.T @ x_

    statistic = _srivastava2011_(n - 1, p, S_)
    p_value = 2 * (1 - norm.cdf(abs(statistic)))

    results = {
        "stat": statistic,
        "p_value": p_value,
    }
    return results
