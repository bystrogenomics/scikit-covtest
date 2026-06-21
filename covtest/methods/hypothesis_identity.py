"""
hypothesis_identity.py
======================

Hypothesis tests for checking whether a covariance matrix is identity.

The primary public entry point for single-sample identity covariance testing is
`identity_covariance_test`.

1. Single-Sample Identity Tests (H0: Sigma = I_p)
--------------------------------------------------
All these tests take a single data matrix X of shape (n_samples, n_features):

Classic / parametric (low-dimensional baseline):
  ledoit_wolf_identity   Ledoit & Wolf (2002)
  nagao_identity         Nagao (1973)

High-dimensional, one-sample, normality-assumed or robust:
  srivastava_2005_identity      Srivastava (2005)
  srivastava2011_single_sample  Srivastava, Kollo & von Rosen (2011)
  srivastava_2014_identity      Srivastava, Yanagihara & Kubokawa (2014)
  tyler_identity                Tyler (1987)
  fisher_single_sample          Fisher (2012) – statistic T2
  test_identity_T2              Ahmad & von Rosen (2015)

High-dimensional, U-statistic / non-normal:
  chen_2010_identity            Chen, Zhang & Zhong (2010)
  xu_2023_identity              Xu et al. (2023 / 2025)

LRT-based:
  one_sample_cov_test           Likelihood Ratio Test

2. Multi-Sample / Common-Covariance Tests (Internal / Advanced)
----------------------------------------------------------------
These tests are not omnibus tests of equality of covariance matrices. They generally
assume covariance homogeneity and evaluate secondary hypotheses:

  ahmad_2017_identity           Assumes Sigma_1 = ... = Sigma_g = Sigma,
                                and tests whether the common Sigma = I_p.

References
----------
Nagao, H. (1973). Annals of Statistics 1(4), 700-709.
Ledoit, O. & Wolf, M. (2002). Annals of Statistics 30(4), 1081-1102.
Srivastava, M. S. (2005). J. Japan Statist. Soc. 35(2), 251-272.
Tyler, D. E. (1987). Annals of Statistics 15(1), 234-251.
Srivastava, M. S., Kollo, T. & von Rosen, D. (2011).
    J. Multivariate Analysis 102, 1090-1103.
Fisher, T. J. (2012). J. Statistical Planning & Inference 142, 312-326.
Ahmad, M. R. & von Rosen, D. (2015).
    Communications in Statistics – Theory and Methods 44(7), 1387-1398.
Chen, S. X., Zhang, L.-X. & Zhong, P.-S. (2010).
    J. American Statistical Association 105(490), 810-819.
Srivastava, M. S., Yanagihara, H. & Kubokawa, T. (2014).
    J. Multivariate Analysis 130, 289-309.
Ahmad, M. R. (2017). Scandinavian J. Statistics 44, 500-523.
Xu, G. et al. (2023 / 2025). Scandinavian J. Statistics 52, 249-269.
"""

import numpy as np
import numpy.linalg as la
import scipy.stats as stats
from numpy.linalg import slogdet
from scipy.stats import norm

from . import _srivastava_2005 as s2005
from . import _tylers as tyler
from . import _ahmad2015 as ahmad2015
from . import _srivastava_yanagihara as sya
from . import _chen_xu_gram as cxg
from .utils import (
    covariance_traces,
    result_dict,
    sample_covariance,
    validate_data_matrix,
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers shared with original functions
# ─────────────────────────────────────────────────────────────────────────────


def _ledoit_wolf_stat(data):
    """Compute the Ledoit–Wolf test statistic W."""
    n, p = data.shape
    sample_cov_matrix, trace_S, _ = covariance_traces(data)
    SmI = sample_cov_matrix - np.eye(p)
    trace_smi2 = np.trace(SmI @ SmI)
    W = 1 / p * trace_smi2 - 1 / (n * p) * trace_S**2 + p / n
    return W


def _nagao_stat(data):
    """Compute Nagao's test statistic V."""
    _, trace_S, trace_S2 = covariance_traces(data)
    p = data.shape[1]
    V = 1 / p * trace_S2 - 2 / p * trace_S + 1
    return V


def _covariance_under_null(S, Sigma):
    if Sigma is None or (isinstance(Sigma, str) and Sigma == "identity"):
        return S

    Sigma = np.asarray(Sigma, dtype=np.float64)
    if Sigma.shape != S.shape:
        raise ValueError("Sigma must have the same shape as S.")

    Sigma = 0.5 * (Sigma + Sigma.T)
    evals, evecs = la.eigh(Sigma)

    if np.any(evals <= 0):
        raise ValueError("Sigma must be positive definite.")

    inv_sqrt = (evecs / np.sqrt(evals)) @ evecs.T
    S0 = inv_sqrt @ S @ inv_sqrt
    return 0.5 * (S0 + S0.T)


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


def _srivastava2011_(n, p, S_):
    term1 = (
        (n**2 / ((n - 1) * (n + 2)))
        * (np.trace(S_ @ S_) - np.trace(S_) ** 2 / n)
        / p
    )
    term2 = 2 * (np.trace(S_) / p)
    return n * (term1 - term2 + 1) / 2


# ─────────────────────────────────────────────────────────────────────────────
# Original tests (unchanged)
# ─────────────────────────────────────────────────────────────────────────────


def test_identity_T2(
    X: np.ndarray,
    center: bool = True,
    calibration: str = "ahmad2015",
    tail: str = "upper",
):
    """Ahmad & von Rosen (2015) T2 test for identity covariance matrix.

    Tests H0: Sigma = I using the T2 statistic based on U-statistics.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The input data matrix.
    center : bool, default=True
        If True, center the data by subtracting column means.
    calibration : {"ahmad2015", "auto", "large_p_small_n", "ratio"}, default="ahmad2015"
        Calibration method for computing the z-statistic.
        Under the null, Ahmad/von Rosen (2015) uses:
            z = (n / 2) * T2
        which is the "ahmad2015" (or "large_p_small_n" or "auto") calibration.
        The "ratio" calibration is an explicit opt-in only, and is not the default.
    tail : {"upper", "two-sided"}, default="upper"
        Whether to calculate upper-tail or two-sided p-value.

    Returns
    -------
    result : dict
        A dictionary containing:
        - 'stat' : float
            The computed test statistic.
        - 'p_value' : float
            The computed p-value.
    """
    X = validate_data_matrix(X)
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

    # When data has been centered by subtracting the sample mean, the raw
    # Ahmad/von Rosen (2015) T2 statistic acquires a positive bias of
    # p / n**2 under H0.  The corresponding z-score bias is p / (2*n).
    # We subtract this finite-sample correction so that (n/2)*T2_corrected
    # is mean-zero under the null for any p/n ratio.
    if center:
        T2 = T2 - p / (n ** 2)

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


def ledoit_wolf_identity(X):
    """Ledoit-Wolf (2002) test for identity covariance matrix.

    Tests H0: Sigma = I.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The input data matrix.

    Returns
    -------
    result : dict
        A dictionary containing:
        - 'stat' : float
            The computed test statistic.
        - 'p_value' : float
            The computed p-value.

    References
    ----------
    Ledoit, O., & Wolf, M. (2002). Annals of Statistics 30(4), 1081-1102.
    """
    X = validate_data_matrix(X)
    n, p = X.shape
    W = _ledoit_wolf_stat(X)
    degree_of_freedom = p * (p + 1) / 2
    stat = n * p / 2 * W
    p_value = 1 - stats.chi2.cdf(stat, degree_of_freedom)
    return result_dict(stat, p_value)


def nagao_identity(X):
    """Nagao's (1973) test for identity covariance matrix.

    Tests H0: Sigma = I.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The input data matrix.

    Returns
    -------
    result : dict
        A dictionary containing:
        - 'stat' : float
            The computed test statistic.
        - 'p_value' : float
            The computed p-value.

    References
    ----------
    Nagao, H. (1973). Annals of Statistics 1(4), 700-709.
    """
    X = validate_data_matrix(X)
    n, p = X.shape
    V = _nagao_stat(X)
    degree_of_freedom = p * (p + 1) / 2
    stat = n * p / 2 * V
    p_value = 1 - stats.chi2.cdf(stat, degree_of_freedom)
    return result_dict(stat, p_value)


def srivastava_2005_identity(X):
    """Srivastava (2005) test for identity covariance matrix.

    High-dimensional test for H0: Sigma = I.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The input data matrix.

    Returns
    -------
    result : dict
        A dictionary containing:
        - 'stat' : float
            The computed test statistic.
        - 'p_value' : float
            The computed p-value.

    References
    ----------
    Srivastava, M. S. (2005). J. Japan Statist. Soc. 35(2), 251-272.
    """
    X = validate_data_matrix(X)
    n = X.shape[0]
    S = sample_covariance(X)
    T_1 = s2005.T_1_stat(S, n)
    z_stat = (n / 2) * T_1
    p_value = 1 - stats.norm.cdf(z_stat)
    return result_dict(z_stat, p_value)


def tyler_identity(X, unknown_mean=False, method="tr"):
    """Tyler's (1987) M-estimator test for identity shape matrix.

    Tests H0: Sigma = I.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The input data matrix.
    unknown_mean : bool, default=False
        If True, robust location is estimated and subtracted from data.
    method : {"tr", "log"}, default="tr"
        Test statistic version to use (trace-based or log-based).

    Returns
    -------
    result : dict
        A dictionary containing:
        - 'stat' : float
            The computed test statistic.
        - 'p_value' : float
            The computed p-value.

    References
    ----------
    Tyler, D. E. (1987). Annals of Statistics 15(1), 234-251.
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
        mean_tr = p * (1 + p / (n - 1))
        var_tr = 4 * c**2
        mean_log = -(p - (n - 1)) * np.log(1 - c) - p
        var_log = -2 * np.log(1 - c) - 2 * c
    else:
        C = tyler.tylers_M(X)
        c = p / n
        T_tr = np.trace(C @ C)
        T_log = slogdet(C)[1]
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


def fisher_single_sample(X, Sigma="identity"):
    """Fisher (2012) T2 test for covariance matrix structure.

    Tests H0: Sigma = Sigma0.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The input data matrix.
    Sigma : {"identity"} or array-like of shape (n_features, n_features), default="identity"
        The covariance matrix under the null hypothesis.

    Returns
    -------
    result : dict
        A dictionary containing:
        - 'stat' : float
            The computed test statistic.
        - 'p_value' : float
            The computed p-value.

    References
    ----------
    Fisher, T. J. (2012). J. Statistical Planning & Inference 142, 312-326.
    """
    X = validate_data_matrix(X)
    p = X.shape[1]
    n = X.shape[0]
    S = sample_covariance(X)
    S_ = _covariance_under_null(S, Sigma)

    statistic = _fisher_2012_stat_(n - 1, p, S_)
    p_value = 2 * (1 - norm.cdf(abs(statistic)))

    return result_dict(statistic, p_value)


def srivastava2011_single_sample(X, Sigma="identity"):
    """Srivastava (2011) test for covariance matrix structure.

    Tests the null hypothesis H0: Sigma = Sigma0, where Sigma0 defaults to 
    the identity matrix (Sigma0 = I_p). Note that Sigma="identity" and 
    Sigma=np.eye(p) are equivalent.

    Rejection is based on large positive values of the test statistic. Thus,
    this function returns an upper-tail p-value. It expects a single sample 
    data matrix X, not multiple groups.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The input data matrix.
    Sigma : {"identity"} or array-like of shape (n_features, n_features), default="identity"
        The covariance matrix under the null hypothesis.

    Returns
    -------
    result : dict
        A dictionary containing:
        - 'stat' : float
            The computed test statistic.
        - 'p_value' : float
            The computed p-value (upper-tail).

    References
    ----------
    Srivastava, M. S., Kollo, T. & von Rosen, D. (2011).
    J. Multivariate Analysis 102, 1090-1103.
    """
    X = validate_data_matrix(X)
    p = X.shape[1]
    n = X.shape[0]
    S = sample_covariance(X)
    S_ = _covariance_under_null(S, Sigma)

    statistic = _srivastava2011_(n - 1, p, S_)
    p_value = float(norm.sf(statistic))

    return result_dict(statistic, p_value)


def one_sample_cov_test(X, mean=None, S=None):
    """LRT-based high-dimensional identity covariance test.

    Tests H0: Sigma = Sigma0.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The input data matrix.
    mean : array-like of shape (n_features,), optional
        The mean vector for adjusting the data. If None, the sample mean is used.
    S : array-like of shape (n_features, n_features), optional
        The covariance matrix under the null hypothesis. If None, the identity matrix
        is assumed.

    Returns
    -------
    result : dict
        A dictionary containing:
        - 'p_value' : float
            The p-value of the test.
        - 'z_value' : float
            The computed Z-value for the test.
        - 'lrt' : float
            The likelihood ratio test statistic.
    """
    X = validate_data_matrix(X)
    n, p = X.shape
    y = p / n
    N = n - 1
    yN = p / N

    if S is not None:
        S_half = la.cholesky(S)
        X = X @ la.inv(S_half)

    if mean is None:
        X = X - np.mean(X, axis=0)
        S_matrix = X.T @ X / N
    else:
        X = X - mean
        S_matrix = X.T @ X / n

    lrt = np.sum(np.diag(S_matrix)) - np.log(la.det(S_matrix)) - p
    mu1 = -0.5 * np.log(1 - y)
    sigma1 = -2 * np.log(1 - y) - 2 * y
    z_value = (lrt - p * (1 + (1 - yN) / yN * np.log(1 - yN)) - mu1) / np.sqrt(
        sigma1
    )
    p_value = norm.sf(z_value)

    return {"p_value": p_value, "z_value": z_value, "lrt": lrt}


def srivastava_2014_identity(X):
    """
    Srivastava, Yanagihara & Kubokawa (2014) test for identity covariance.

    Tests H₀: Σ = Iₚ using the statistic

        T₂ = (n/2) · (â₂ − 2â₁ + 1)

    where â₂ is the new **unbiased O(N²)** estimator of a₂ = tr(Σ²)/p
    (equation 2.5), valid under a general class of distributions (no
    normality required), and â₁ = tr(S)/p.

    Under H₀: T₂ →ᴅ N(0, 1)  (Corollary 3.2 of the paper).

    The key improvement over Srivastava (2005) is that â₂ is unbiased
    for the full class of distributions in model (1.1)–(1.3), whereas the
    earlier â₂ₛ is biased when K₄ ≠ 0 (non-Gaussian kurtosis).

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data matrix. Centering is performed internally.

    Returns
    -------
    result : dict  with keys 'stat' (T₂) and 'p_value' (upper-tail).

    References
    ----------
    Srivastava, M. S., Yanagihara, H., & Kubokawa, T. (2014).
    "Tests for covariance matrices in high dimension with less sample size."
    J. Multivariate Analysis, 130, 289–309.
    https://doi.org/10.1016/j.jmva.2014.06.003
    """
    X = validate_data_matrix(X)
    N, p = X.shape
    if N < 4:
        raise ValueError("Srivastava (2014) test requires N >= 4.")
    n = N - 1  # paper's n = N − 1

    a1 = sya.a_1_hat(X)
    a2 = sya.a_2_hat(X)

    T2 = (n / 2.0) * (a2 - 2.0 * a1 + 1.0)
    p_value = float(norm.sf(T2))
    return result_dict(T2, p_value)


def chen_2010_identity(X):
    """
    Chen, Zhang & Zhong (2010) test for identity covariance matrix.

    Tests H₀: Σ = Iₚ using the statistic

        Vₙ = (1/p)·T₂,ₙ − (2/p)·T₁,ₙ + 1

    where T₁,ₙ (= tr(S)) and T₂,ₙ are **location-invariant U-statistic**
    estimators of tr(Σ) and tr(Σ²) respectively (Theorem 2 of the paper):

        (n/2)·Vₙ →ᴅ N(0, 1)  under H₀.

    This test is nonparametric: it does not assume a specific parametric
    distribution and accommodates p >> n.  The test rejects H₀ for large
    values of the standardised statistic (upper-tail).

    The key difference from Srivastava (2005) is the 4th-order U-statistic
    estimator of tr(Σ²), which is unbiased under mild moment conditions and
    does not require normality.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data matrix (n ≥ 4 required). Centering is performed internally.

    Returns
    -------
    result : dict  with keys 'stat' ((n/2)·Vₙ) and 'p_value' (upper-tail).

    References
    ----------
    Chen, S. X., Zhang, L.-X., & Zhong, P.-S. (2010).
    "Tests for High-Dimensional Covariance Matrices."
    J. American Statistical Association, 105(490), 810-819.
    https://doi.org/10.1198/jasa.2010.tm09560
    """
    X = validate_data_matrix(X)
    n, p = X.shape
    if n < 4:
        raise ValueError(
            f"chen_2010_identity requires n ≥ 4 (got n={n}). "
            "The 4th-order U-statistic T₂ is not defined for n < 4."
        )

    blk = cxg.gram_blocks(X)
    T1 = cxg.T1_chen(blk)  # = tr(S), estimator of tr(Σ)
    T2 = cxg.T2_chen(blk)  # unbiased estimator of tr(Σ²)

    Vn = T2 / p - 2.0 * T1 / p + 1.0
    stat = (n / 2.0) * Vn
    p_value = float(norm.sf(stat))
    return result_dict(stat, p_value)


def xu_2023_identity(X):
    """
    Xu et al. (2023) elliptical-adjusted identity covariance test.

    Tests H₀: Σ = Iₚ using the kurtosis-corrected statistic

        V̂ₙ,ₚ = p · (T₂/p − 2T₁/p + 1) / σ̂₀,ₙ,ₚ

    where σ̂²₀,ₙ,ₚ (equation 14) is estimated from a 5th-order U-statistic
    (δ̂ₙ,ₚ) and corrects for the elliptical-family high-order coordinate
    dependence that invalidates the Chen/Ahmad tests under non-Gaussian
    elliptical distributions.

    Under H₀: V̂ₙ,ₚ →ᴅ N(0, 1)  (Theorem 3).

    The correction vanishes for Gaussian data (δ̂/T₃ → (p+2)/p), in which
    case V̂ₙ,ₚ reduces to the Chen (2010) test statistic (n/2)·Vₙ.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data matrix (n ≥ 5 required). Centering is performed internally.

    Returns
    -------
    result : dict  with keys 'stat' (V̂ₙ,ₚ) and 'p_value' (upper-tail).

    References
    ----------
    Xu, G., Zhou, Y., et al. (2023 / 2025).
    "Adjusted location-invariant U-tests for the covariance matrix with
    elliptically high-dimensional data."
    Scandinavian Journal of Statistics, 52, 249-269.
    https://doi.org/10.1111/sjos.12738
    """
    X = validate_data_matrix(X)
    n, p = X.shape
    if n < 5:
        raise ValueError(
            f"xu_2023_identity requires n ≥ 5 (got n={n}). "
            "The 5th-order U-statistic δ̂ is not defined for n < 5."
        )

    blk = cxg.gram_blocks(X)

    T1 = cxg.T1_chen(blk)  # tr(S), estimator of tr(Σ)
    T2 = cxg.T2_chen(blk)  # estimator of tr(Σ²)
    T3 = cxg.T3_xu(blk)  # estimator of tr²(Σ)
    delta = cxg.delta_hat_xu(blk)  # 5th-order U-statistic

    sigma2 = cxg.sigma2_hat_xu(delta, T3, n, p)

    # Raw distance: (T₂ − 2T₁ + p) / p  =  T₂/p − 2T₁/p + 1
    raw = T2 / p - 2.0 * T1 / p + 1.0
    stat = p * raw / np.sqrt(sigma2)
    p_value = float(norm.sf(stat))
    return result_dict(stat, p_value)


def ahmad_2017_identity(Xs):
    """
    Ahmad (2017) multi-sample test for a common identity covariance matrix.

    .. warning::
       [INTERNAL / ADVANCED] This method is not an omnibus test of H0: Sigma_i = I
       under heterogeneous alternatives. It requires g >= 2 and assumes a common 
       covariance matrix across groups (covariance homogeneity), testing whether 
       this common covariance is identity.

    The test statistic is (equations 5 and 22–23 of the paper):

        T_b = C₃/p − 2·C₁/p + 1

    where
        C₁ = Q-weighted average of Bᵢ = tr(Sᵢ)  (estimator of tr(Σ))
        C₂ = 1/[g(g−1)] Σᵢ≠ⱼ BᵢBⱼ             (estimator of [tr(Σ)]²)
        C₃ = Σᵢ≠ⱼ P(nᵢ,nⱼ)·Bᵢⱼ / P*           (estimator of tr(Σ²))

    and Bᵢ = tr(Sᵢ) (trace of sample covariance of population i),
    Bᵢⱼ = tr(SᵢSⱼ) (cross-trace of sample covariances).

    Under H₀ (Theorem 5 of the paper):  Values standardise to a standard normal:
    stat = sqrt(P*/4) * T_b -> N(0, 1)

    where P* = Σᵢ≠ⱼ Q(nᵢ)·Q(nⱼ) and Q(nᵢ) = nᵢ(nᵢ-1).

    The two-sample (g=2) and multi-sample (g≥2) cases are handled by the
    same formula; for g=2 this reduces to equations (13)–(15) of the paper.

    Parameters
    ----------
    Xs : list of array-like, each of shape (nᵢ, p)
        Data matrices from g ≥ 2 populations. All must have the same p.

    Returns
    -------
    result : dict  with keys 'stat' (standardised T_b) and 'p_value' (upper-tail).

    References
    ----------
    Ahmad, M. R. (2017).
    "Location-invariant Multi-sample U-tests for Covariance Matrices
    with Large Dimension."
    Scandinavian Journal of Statistics, 44, 500-523.
    https://doi.org/10.1111/sjos.12262
    """
    if len(Xs) < 2:
        raise ValueError(
            "ahmad_2017_identity requires at least 2 samples (g ≥ 2)."
        )

    g = len(Xs)
    arrays = [validate_data_matrix(np.asarray(X, dtype=np.float64)) for X in Xs]
    p = arrays[0].shape[1]
    if not all(A.shape[1] == p for A in arrays):
        raise ValueError("All samples must have the same number of features p.")

    # ── per-sample quantities ─────────────────────────────────────────────
    ns = [A.shape[0] for A in arrays]  # sample sizes nᵢ
    Qs = [ni * (ni - 1) for ni in ns]  # Q(nᵢ) = nᵢ(nᵢ-1)

    # Sample covariances (p×p)
    Ss = []
    for A in arrays:
        Ac = A - A.mean(axis=0)
        Ss.append(Ac.T @ Ac / (A.shape[0] - 1))

    # Bᵢ = tr(Sᵢ)  (unbiased estimator of tr(Σᵢ))
    Bs = [float(np.trace(Si)) for Si in Ss]

    # ── pooled estimators ─────────────────────────────────────────────────
    sum_Q = sum(Qs)
    C1 = sum(Qs[i] * Bs[i] for i in range(g)) / sum_Q

    P_star_Q = 0.0
    C3_num = 0.0
    for i in range(g):
        for j in range(g):
            if i != j:
                Pij = float(Qs[i] * Qs[j])  # P(nᵢ,nⱼ) = Q(nᵢ)·Q(nⱼ)
                Bij = float(np.trace(Ss[i] @ Ss[j]))
                P_star_Q += Pij
                C3_num += Pij * Bij
    C3 = C3_num / P_star_Q

    T_b = C3 / p - 2.0 * C1 / p + 1.0
    P_n = float(
        sum(ns[i] * ns[j] for i in range(g) for j in range(g) if i != j)
    )
    stat = float(np.sqrt(P_n / 4.0) * T_b)
    p_value = float(norm.sf(stat))
    return result_dict(stat, p_value)


def identity_covariance_test(X, method="chen_2010", **kwargs):
    """Clean public wrapper for single-sample identity covariance test.

    Tests the null hypothesis:
        H0: Sigma = I_p
    for a single sample data matrix X, against non-identity covariance alternatives.

    Note: This is NOT a test for equality of covariance matrices across multiple groups.
    For multi-sample tests, please refer to the two-sample or proportionality testing modules.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The input data matrix.
    method : str, default="chen_2010"
        The test method to apply. Must be one of:
        - "chen_2010": Chen, Zhang & Zhong (2010) high-dimensional U-statistic test.
        - "ahmad2015": Ahmad & von Rosen (2015) T2 test using U-statistics.
        - "xu_2023": Xu et al. (2023) elliptically-adjusted identity test.
        - "srivastava_2005": Srivastava (2005) high-dimensional test.
        - "srivastava_2011": Srivastava, Kollo & von Rosen (2011) test.
        - "srivastava_2014": Srivastava, Yanagihara & Kubokawa (2014) test.
        - "ledoit_wolf": Ledoit & Wolf (2002) test.
        - "nagao": Nagao (1973) test.
        - "tyler": Tyler (1987) distribution-free test.
        - "fisher": Fisher (2012) test.
        - "lrt": Likelihood ratio test.
    **kwargs : dict
        Additional arguments passed to the underlying test function.

    Returns
    -------
    result : dict
        A dictionary containing:
        - 'stat' : float
            The standardized test statistic.
        - 'p_value' : float
            The computed p-value (upper-tail).

    Examples
    --------
    >>> import numpy as np
    >>> X = np.random.normal(size=(100, 30))
    >>> result = identity_covariance_test(X)
    >>> print(result["stat"], result["p_value"])
    """
    methods_map = {
        "chen_2010": chen_2010_identity,
        "ahmad2015": test_identity_T2,
        "xu_2023": xu_2023_identity,
        "srivastava_2005": srivastava_2005_identity,
        "srivastava_2011": srivastava2011_single_sample,
        "srivastava_2014": srivastava_2014_identity,
        "ledoit_wolf": ledoit_wolf_identity,
        "nagao": nagao_identity,
        "tyler": tyler_identity,
        "fisher": fisher_single_sample,
        "lrt": one_sample_cov_test,
    }

    if method not in methods_map:
        raise ValueError(
            f"Unknown method '{method}'. Must be one of: {list(methods_map.keys())}"
        )

    res = methods_map[method](X, **kwargs)

    stat = res.get("stat", res.get("z_value", res.get("lrt", np.nan)))
    p_value = res.get("p_value", np.nan)

    return {
        "stat": float(stat),
        "p_value": float(p_value),
    }

