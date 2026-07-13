from typing import Dict, Literal, Optional

import numpy as np
import numpy.linalg as la
import scipy.stats as stats
from numpy.linalg import inv, slogdet
from scipy.integrate import quad

from . import _cheng_2019 as cheng2019
from . import _eriksen_1987 as eriksen1987
from . import _liu_2014 as liu2014
from . import _tsukuda_2019 as tsukuda2019
from . import _ahmad as ahmad2022
from .utils import validate_data_matrix, validate_matching_data_matrices

ArrayLike = np.ndarray


################
# Eriksen 1987 #
################


def bootstrap_bartlett_adjusted_proportionality_test(
    X,
    Y,
    ridge=0.0,
    tol=1e-10,
    max_iter=10_000,
    B=400,
    refit_mle_each_boot=True,
    random_state=None,
):
    """
    Bootstrap Bartlett-adjusted Wilks LRT for proportional covariance matrices.

    This function performs a parametric-bootstrap Bartlett-style calibration of the
    likelihood ratio test statistic. Note that this is NOT Eriksen's closed-form analytic
    Bartlett adjustment (Theorem 6.1 of Eriksen 1987).

    Parameters
    ----------
    X : array-like of shape (n_samples_1, n_features)
        First sample data matrix.
    Y : array-like of shape (n_samples_2, n_features)
        Second sample data matrix.
    ridge : float, default 0.0
        Small diagonal ridge to improve numerical stability when needed.
    tol, max_iter : solver settings for the H0 MLE.
    B : int, default 400
        Number of parametric bootstrap replicates to estimate the Bartlett factor.
    refit_mle_each_boot : bool, default True
        If True, re-fit (hat(Sigma), hat{c}) inside each bootstrap replicate.
        Highly recommended. If False, the null model is not re-maximized, which
        means it is not a true LRT bootstrap calibration.
    random_state : int or np.random.Generator, optional
        RNG seed or Generator.
    """
    X = validate_data_matrix(X)
    Y = validate_data_matrix(Y)
    return eriksen1987.bootstrap_bartlett_adjusted_proportionality_test(
        X_groups=[X, Y],
        ridge=ridge,
        tol=tol,
        max_iter=max_iter,
        B=B,
        refit_mle_each_boot=refit_mle_each_boot,
        random_state=random_state,
    )


def bartlett_adjusted_proportionality_test(*args, **kwargs):
    import warnings

    warnings.warn(
        "bartlett_adjusted_proportionality_test is deprecated; "
        "use bootstrap_bartlett_adjusted_proportionality_test instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return bootstrap_bartlett_adjusted_proportionality_test(*args, **kwargs)


def proportionality_test_LZ(X, Y, regularize=0.0):
    r"""
    Liu–Xu–Zheng–Tian (2014) proportionality test for two covariance matrices.

    Parameters
    ----------
    X : array, shape (N1, p)
        Sample 1 (rows are observations).
    Y : array, shape (N2, p)
        Sample 2 (rows are observations).
    regularize : float, default 0.0
        Optional ridge parameter. Must be 0.0. Retained only for backward compatibility.

    Returns
    -------
    result : dict
        A dictionary containing:
        - 'stat' : float
            The standardized Liu 2014 test statistic Z.
        - 'p_value' : float
            The one-sided upper-tail p-value.
    Notes
    -----
    Requires p < n2 = N2 - 1.
    """
    X = validate_data_matrix(X)
    Y = validate_data_matrix(Y)
    N1, p = X.shape
    N2, p2 = Y.shape
    if p != p2:
        raise ValueError("X and Y must have the same number of columns (p).")

    if regularize != 0.0:
        raise ValueError("regularize is not supported and must be 0.0.")

    n1 = N1 - 1
    n2 = N2 - 1
    if not (p < n2):
        raise ValueError(f"Requirement p < n2 (={n2}) violated (p={p}).")

    # Unbiased sample covariances
    S1 = np.cov(X.T)
    S2 = np.cov(Y.T)

    S2_inv = la.inv(S2)
    A = S1 @ S2_inv

    trA = np.trace(A)
    trA2 = np.trace(A @ A)
    Tn = (p**2) * (trA2 / (trA**2)) - p

    # Ratios and h
    y1 = p / n1
    y2 = p / n2
    h = np.sqrt(y1 + y2 - y1 * y2)

    # Kurtosis parameters (estimated)
    beta_x = liu2014._beta_hat(X)
    beta_y = liu2014._beta_hat(Y)

    # Asymptotic mean and variance pieces
    mu_Tn = (h**2 + y2**2) / (1 - y2) ** 2 + beta_x * y1 + beta_y * y2
    v_Tn = 4 * h**2 * (h**2 + 2 * y2**2) / (1 - y2) ** 4

    # Z-score (one-sided test: large positive values reject H0)
    Z = (Tn - (mu_Tn + p * h**2 / (1 - y2))) / np.sqrt(v_Tn)

    p_one = stats.norm.sf(Z)

    return {"stat": float(Z), "p_value": float(p_one)}


def proportionality_test_signs(
    X: ArrayLike,
    Y: ArrayLike,
    center: Literal["spatial_median", "mean"] = "spatial_median",
    calibration: Literal[
        "permutation", "asymp_spherical", "asymp_empirical"
    ] = "permutation",
    n_perm: int = 999,
    random_state: Optional[int] = None,
) -> Dict[str, float]:
    """
    Cheng Robust high-dimensional test of proportional covariance via spatial signs.

    This tests H0: Sigma_1 = c Sigma_2 for some c > 0 by equivalently testing H0: S1 = S2,
    where S is the spatial sign covariance. The test statistic is
        T' = p * (A + B - 2 C),
    with
        A = mean_{i != i'} (u_i^T u_{i'})^2,
        B = mean_{j != j'} (v_j^T v_{j'})^2,
        C = mean_{i, j}     (u_i^T v_j)^2,
    and u_i, v_j are unit-length spatial signs after centering.

    Calibration options:
    - 'permutation': label-exchange permutation p-values computed from a pooled Gram matrix.
      This controls type I error when the two sign-distributions are exchangeable under H0.
    - 'asymp_spherical': normal approximation with a closed-form variance that is exact for
      spherical directions and often reasonable at large p.
    - 'asymp_empirical': normal approximation with an empirical plug-in variance computed
      from sample variances of pairwise squared cosines (ignores pair dependence).

    Parameters
    ----------
    X : ndarray of shape (n1, p)
        Sample 1, rows are observations.
    Y : ndarray of shape (n2, p)
        Sample 2, rows are observations.
    center : {'spatial_median', 'mean'}, default 'spatial_median'
        Centering used before taking spatial signs.
    calibration : {'permutation', 'asymp_spherical', 'asymp_empirical'}, default 'permutation'
        How to obtain p-values.
    n_perm : int, default 999
        Number of permutations for the 'permutation' calibration.
    random_state : int or None, default None
        Random seed for permutation.

    Returns
    -------
    result : dict
        Keys include:
        - 'Tn'              : observed test statistic T'
        - 'pvalue_two_sided': two-sided p-value
        - 'pvalue_right'    : right-tail p-value for T' > 0
        - 'n1','n2','p'     : sample sizes and dimension
        - 'calibration'     : calibration method used
        - 'Z'               : normal Z-score if applicable, else np.nan

    Notes
    -----
    This implementation avoids matrix inversion and works when p > max(n1, n2).
    For heavy-tailed elliptical data, spatial signs provide robustness compared to
    likelihood or inverse-covariance based tests.

    The permutation calibration tests equality of sign-distributions, which is
    slightly stronger than proportionality unless elliptical assumptions hold.
    Use asymptotic calibrations when permutation is too costly or when exchangeability
    is not appropriate, but expect some approximation error in small samples.
    """
    X = validate_data_matrix(X)
    Y = validate_data_matrix(Y)
    n1, p1 = X.shape
    n2, p2 = Y.shape
    if p1 != p2:
        raise ValueError(
            "X and Y must have the same number of columns (features)."
        )
    p = p1

    # Spatial signs
    U, kept1 = cheng2019.spatial_signs(X, center=center)
    V, kept2 = cheng2019.spatial_signs(Y, center=center)
    n1_eff, n2_eff = U.shape[0], V.shape[0]
    if n1_eff < 2 or n2_eff < 2:
        raise ValueError(
            "After centering, each group must have at least 2 non-degenerate rows."
        )

    # Pool and build Gram matrix once
    W = np.vstack([U, V])  # shape (n1_eff + n2_eff, p)
    G = W @ W.T  # shape (n, n), entries are pairwise cosines

    idx1 = np.arange(n1_eff)
    idx2 = np.arange(n1_eff, n1_eff + n2_eff)

    # Observed test statistic
    Tn = cheng2019._T_from_gram(G, idx1, idx2, p)

    # Calibration
    Z = np.nan
    if calibration == "permutation":
        rng = np.random.default_rng(random_state)
        _, p_right = cheng2019._perm_pvalue_from_gram(
            G, n1_eff, n2_eff, p, n_perm, rng, Tn
        )
        Z = 0
    elif calibration == "asymp_spherical":
        v0 = cheng2019._var_spherical_asymp(n1_eff, n2_eff, p)
        sd = np.sqrt(max(v0, 0.0))
        Z = Tn / sd if sd > 0 else np.nan
        p_right = 1.0 - cheng2019._phi(Z) if np.isfinite(Z) else np.nan
    elif calibration == "asymp_empirical":
        v0 = cheng2019._var_empirical_from_gram(G, idx1, idx2, p)
        sd = np.sqrt(max(v0, 0.0))
        Z = Tn / sd if sd > 0 else np.nan
        p_right = 1.0 - cheng2019._phi(Z) if np.isfinite(Z) else np.nan
    else:
        raise ValueError("Unknown calibration method.")

    return {
        "p_value": float(p_right),
        "stat": float(Z) if np.isfinite(Z) else np.nan,
    }


def proportionality_plrt(X, Y, dist_moments="gaussian"):
    """
    Pseudo-Likelihood Ratio Test (PLRT) for proportionality of two covariance matrices.

    Parameters
    ----------
    X : ndarray, shape (n1, p)
        Sample 1 (rows are observations).
    Y : ndarray, shape (n2, p)
        Sample 2 (rows are observations).
    dist_moments : str, default="gaussian"
        Distributional assumption to set 4th moment adjustments.
        - "gaussian": sets beta_x = beta_y = 0.

    Returns
    -------
    result : dict
        Dictionary containing:
        - T1 : PLRT statistic
        - Z : standardized test statistic ~ N(0,1) under H0
        - pvalue_one_sided
        - pvalue_two_sided
        - mu_T1, sigma2_T1
    """
    n1, p = X.shape
    n2, p2 = Y.shape
    assert p == p2, "Both groups must have same dimension"

    # Sample covariances (unbiased)
    S1 = np.cov(X, rowvar=False, bias=False)
    S2 = np.cov(Y, rowvar=False, bias=False)

    # Ratios
    y1 = p / n1
    y2 = p / n2
    h = np.sqrt(y1 + y2 - y1 * y2)

    # kurtosis parameters
    if dist_moments == "gaussian":
        beta_x, beta_y = 0.0, 0.0
    else:
        raise NotImplementedError("Only Gaussian case implemented.")

    # T1 statistic
    X = validate_data_matrix(X)
    Y = validate_data_matrix(Y)
    A = S1 @ inv(S2)
    tr_term = np.trace(A) / p
    sign, logdet = slogdet(A)
    if sign <= 0:
        raise ValueError("Matrix product S1*S2^{-1} not positive definite.")
    T1 = p * np.log(tr_term) - logdet

    # Compute LSD bounds
    a = (1 - h) ** 2 / (1 - y2) ** 2
    b = (1 + h) ** 2 / (1 - y2) ** 2

    def f_density(x, y1, y2):
        if x < a or x > b:
            return 0.0
        return (
            (1 - y2)
            * np.sqrt((b - x) * (x - a))
            / (2 * np.pi * x * (y1 + y2 * x))
        )

    # compute a0 and a1 integrals
    def integrand_a0(x):
        return x * f_density(x, y1, y2)

    def integrand_a1(x):
        return np.log(x) * f_density(x, y1, y2)

    a0, _ = quad(integrand_a0, a, b, limit=200)
    a1, _ = quad(integrand_a1, a, b, limit=200)

    # mu_CLT vector
    mu_CLT = np.array(
        [
            y2 / (1 - y2) ** 2 + (y2 * beta_y) / (1 - y2),
            0.5 * np.log((1 - h**2) / (1 - y2) ** 2)
            - 0.5 * beta_x * y1
            + 0.5 * beta_y * y2,
        ]
    )

    # Sigma_CLT matrix
    Sigma_CLT = np.array(
        [
            [
                2 * h**2 / (1 - y2) ** 4
                + (beta_x * y1 + beta_y * y2) / (1 - y2) ** 2,
                (beta_x * y1 + beta_y * y2) / (1 - y2)
                + 2 * h**2 / (1 - y2) ** 2,
            ],
            [
                (beta_x * y1 + beta_y * y2) / (1 - y2)
                + 2 * h**2 / (1 - y2) ** 2,
                -2 * np.log(1 - h**2) + beta_x * y1 + beta_y * y2,
            ],
        ]
    )

    v = np.array([1 / a0, -1])
    mu_T1 = v @ mu_CLT
    sigma2_T1 = v @ Sigma_CLT @ v

    # Standardize
    Z = (T1 - mu_T1 - p * (np.log(a0) - a1)) / np.sqrt(sigma2_T1)
    p_one = 1 - stats.norm.cdf(Z)

    return {
        "stat": float(Z),
        "p_value": float(p_one),
    }


# Check
def proportional_cov_test_tsukuda(
    X: np.ndarray, Y: np.ndarray, single_side: bool = True
) -> Dict[str, float]:
    """Tsukuda & Matsuura (2019) test of proportional covariance matrices.

    Tests H0: Sigma_x = k * Sigma_y for some k > 0 against H1: Sigma_x is not proportional to Sigma_y.

    Parameters
    ----------
    X : array-like of shape (n_samples_1, n_features)
        First sample data matrix.
    Y : array-like of shape (n_samples_2, n_features)
        Second sample data matrix.
    single_side : bool, default=True
        If True, returns one-sided upper-tail p-value; otherwise two-sided.

    Returns
    -------
    result : dict
        A dictionary containing:
        - 'stat' : float
            The standardized test statistic.
        - 'p_value' : float
            The computed p-value.
    """
    X, Y = validate_matching_data_matrices(X, Y)
    m, p = X.shape
    n, _ = Y.shape
    if m < 3 or n < 3:
        raise ValueError(
            "Each group must have at least 3 observations (for Tsukuda-Matsuura a2 estimator)."
        )

    # Sample covariances with ddof=1
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    Sx = (Xc.T @ Xc) / (m - 1)
    Sy = (Yc.T @ Yc) / (n - 1)

    a_x1 = np.trace(Sx) / p
    a_y1 = np.trace(Sy) / p
    a_xy = np.trace(Sx @ Sy) / p

    a_x2 = tsukuda2019._a2_hat_tsukuda(X)
    a_y2 = tsukuda2019._a2_hat_tsukuda(Y)

    # Core statistic and variance estimate
    T = (m * n / (m + n)) * (
        a_x2 / (a_x1**2) + a_y2 / (a_y1**2) - 2.0 * a_xy / (a_x1 * a_y1)
    )

    b2_hat = ((m**2) / (m**2 + n**2)) * (a_x2 / (a_x1**2)) + (
        (n**2) / (m**2 + n**2)
    ) * (a_y2 / (a_y1**2))

    Z = T / (2.0 * b2_hat) if b2_hat > 0 else np.nan

    if single_side:
        # Note: the paper proposes the one-sided upper-tail test.
        p_value = 1.0 - stats.norm.cdf(Z)
    else:
        p_value = 2.0 * stats.norm.sf(abs(Z))

    return {"stat": float(Z), "p_value": float(p_value)}


def ahmad_2022_proportionality_test(X, Y, alternative="two-sided"):
    """Ahmad (2022) proportionality test of two covariance matrices.

    Tests H0: Sigma_1 = delta * Sigma_2 for some delta > 0.

    Parameters
    ----------
    X : array-like of shape (n_samples_1, n_features)
        First sample data matrix.
    Y : array-like of shape (n_samples_2, n_features)
        Second sample data matrix.
    alternative : {"two-sided", "greater", "less"}, default="two-sided"
        Alternative hypothesis formulation.

    Returns
    -------
    result : dict
        A dictionary containing:
        - 'stat' : float
            The standardized test statistic.
        - 'p_value' : float
            The computed p-value.
        - 'T_hat' : float
            The unstandardized test statistic.
        - 'z' : float
            Identical to 'stat'.
        - 'E1', 'E2', 'E12' : floats
            Trace estimates.
    """
    X, Y = validate_matching_data_matrices(X, Y)
    n1, p = X.shape
    n2, _ = Y.shape

    E1 = ahmad2022.estimate_Ei_trSigma2(X)
    E2 = ahmad2022.estimate_Ei_trSigma2(Y)
    E12 = ahmad2022.estimate_E12_trSigma1Sigma2(X, Y)

    if E1 <= 0 or E2 <= 0:
        raise ValueError(
            f"Non-positive E1 or E2 encountered (E1={E1}, E2={E2}). "
            "This can happen with very small n or numerical issues."
        )

    T_hat = (E12 / np.sqrt(E1 * E2)) - 1.0
    z = 0.5 * np.sqrt(n1 * n2) * T_hat

    if alternative == "two-sided":
        p_value = 2.0 * (1.0 - stats.norm.cdf(abs(z)))
    elif alternative == "greater":
        p_value = 1.0 - stats.norm.cdf(z)
    elif alternative == "less":
        p_value = stats.norm.cdf(z)
    else:
        raise ValueError(
            "alternative must be one of: 'two-sided', 'greater', 'less'."
        )

    return {
        "stat": float(z),
        "p_value": float(p_value),
    }
