from typing import Dict, Literal, Optional

import numpy as np
import numpy.linalg as la
import scipy.stats as stats
from numpy.linalg import inv, slogdet
from scipy.integrate import quad
from scipy.stats import chi2

from . import _cheng_2019 as cheng2019
from . import _eriksen_1987 as eriksen1987
from . import _flurry_1986 as flurry1986
from . import _liu_2014 as liu2014
from . import _tsukuda_2019 as tsukuda2019

ArrayLike = np.ndarray

##########
# Flurry #
##########


def flury_proportionality_test(
    X_in,
    Y,
    max_iter: int = 1000,
    tol: float = 1e-9,
    ridge: float = 0.0,
    init_c: Optional[np.ndarray] = None,
):
    """
    Convenience wrapper that accepts raw group data matrices.

    Parameters
    ----------
    X_list : list of arrays
        Each X_i has shape (n_i, p). This function computes S_i = scatter / n_i.
    Returns
    -------
    out : dict
        See flury_proportionality_test_from_cov.
    """
    X_list = [X_in, Y]
    S_list = []
    n_list = []
    for X in X_list:
        X = np.asarray(X, float)
        n_i, p_i = X.shape
        if not S_list:
            p = p_i
        elif p_i != p:
            raise ValueError(
                "All groups must have the same number of variables p."
            )
        S_list.append(flurry1986._cov_mle(X))
        n_list.append(n_i)
    return flurry1986.flury_proportionality_test_from_cov(
        S_list, n_list, max_iter=max_iter, tol=tol, ridge=ridge, init_c=init_c
    )


################
# Eriksen 1987 #
################


def bartlett_adjusted_proportionality_test(
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
    Bartlett-adjusted Wilks LRT for proportional covariance matrices.

    Strategy
    --------
    1) Fit H0 by MLE (hat(Sigma), hat{c}) using the proportional model Sigma_k = c_k Sigma.
    2) Compute the unadjusted Wilks statistic T = -2 log Λ from the data.
    3) Parametric Bartlett factor:
         simulate B datasets under H0 with the fitted hat(Sigma)_k =
         hat{c}_k hat(Sigma) and the same n_k;
         compute T_b for each; set phi = df / mean(T_b).
       The adjusted statistic is T_adj = phi * T, yielding E[T_adj]
       approx df.

    Parameters
    ----------
        X_groups : list of arrays, optional
        Observations per group (N_k x p). If given, S_list and n_list are
        built with ddof=1.
    ridge : float, default 0.0
        Small diagonal ridge to improve numerical stability when needed.
    tol, max_iter : solver settings for the H0 MLE.
    B : int, default 400
        Number of parametric bootstrap replicates to estimate the
        Bartlett factor.
    refit_mle_each_boot : bool, default True
        If True, re-fit (hat(Sigma), hat{c}) inside each bootstrap
        replicate before computing T_b.
        This is more accurate; set False for speed (uses the original
        hat(Sigma), hat{c} in T_b).
    random_state : int or np.random.Generator, optional
        RNG seed or Generator.

    Returns
    -------
    result : dict
        Keys include:
        - 'stat'        : unadjusted Wilks statistic
        - 'df'          : chi-square degrees of freedom
        - 'pvalue'      : unadjusted p-value (χ²_df)
        - 'phi'         : multiplicative Bartlett factor (approx df /
                             E[T] under H0)
        - 'stat_adj'    : Bartlett-adjusted statistic  (phi * stat)
        - 'pvalue_adj'  : Bartlett-adjusted p-value    (χ²_df right-tail)
        - 'Sigma_hat', 'c_hat', 'converged', 'iterations', 'n_list', 'S_list'
    """
    X_groups = [X, Y]
    rng = np.random.default_rng(random_state)

    # 1) Unadjusted fit and statistic
    base = eriksen1987.test_cov_proportionality(
        X_groups=X_groups,
        S_list=None,
        n_list=None,
        ridge=ridge,
        tol=tol,
        max_iter=max_iter,
    )
    Sigma_hat = base["Sigma_hat"]
    c_hat = base["c_hat"]
    S_list = base["S_list"]
    n_list = np.asarray(base["n_list"], dtype=float)
    K = len(S_list)
    p = S_list[0].shape[0]
    df = base["df"]
    T_obs = base["stat"]

    # 2) Build Cholesky factors for simulation under H0
    chol_list = []
    for k in range(K):
        Sk = c_hat[k] * Sigma_hat
        Sk = eriksen1987._ensure_pd(Sk, ridge if ridge > 0 else 1e-12)
        chol_list.append(la.cholesky(Sk))

    # 3) Parametric bootstrap to estimate E[T] under H0 and phi
    T_boot = np.empty(B, dtype=float)
    for b in range(B):
        S_b = []
        for k in range(K):
            n_k = int(n_list[k])
            # Simulate n_k + 1 observations so that ddof=1 yields n_k in Wishart
            Xk = rng.standard_normal((n_k + 1, p)) @ chol_list[k].T
            S_b.append(eriksen1987._sample_cov(Xk))
        if refit_mle_each_boot:
            Sigma_b, c_b, _, _ = eriksen1987.fit_proportional_covariances(
                S_b, n_list, tol=tol, max_iter=max_iter, ridge=ridge
            )
            T_b = eriksen1987._wilks_stat_from_S(
                S_b, n_list, Sigma_b, c_b, ridge=ridge
            )
        else:
            T_b = eriksen1987._wilks_stat_from_S(
                S_b, n_list, Sigma_hat, c_hat, ridge=ridge
            )
        T_boot[b] = T_b

    mean_T = float(np.mean(T_boot))
    # Multiplicative Bartlett factor so that E[phi*T] ≈ df
    phi = df / mean_T if mean_T > 0 else 1.0

    T_adj = phi * T_obs
    p_adj = 1.0 - chi2.cdf(T_adj, df)

    return {
        **base,
        "phi": float(phi),
        "stat": float(T_adj),
        "p_value": float(p_adj),
    }


def proportionality_test_LZ(X, Y, regularize=0.0):
    """
    Liu–Xu–Zheng–Tian (2014) proportionality test for two covariance matrices.

    Parameters
    ----------
    X : array, shape (N1, p)
        Sample 1 (rows are observations).
    Y : array, shape (N2, p)
        Sample 2 (rows are observations).
    regularize : float, default 0.0
        Optional ridge added to \hat\Sigma_2 for numerical stability.

    Returns
    -------
    result : dict with keys
        'Tn'         : test statistic
        'mu_Tn'      : asymptotic mean adjustment (without the + p*h^2/(1-y2) term)
        'v_Tn'       : asymptotic variance
        'Z'          : standardized statistic ~ N(0,1) under H0
        'pvalue_one_sided' : 1 - Phi(Z)
        'pvalue_two_sided' : 2 * (1 - Phi(|Z|))
        'y1','y2','h','beta_x','beta_y'
    Notes
    -----
    Requires p < n2 = N2 - 1.
    """
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    N1, p = X.shape
    N2, p2 = Y.shape
    assert p == p2, "X and Y must have the same number of columns (p)."

    n1 = N1 - 1
    n2 = N2 - 1
    if not (p < n2):
        raise ValueError(f"Requirement p < n2 (={n2}) violated (p={p}).")

    # Unbiased sample covariances
    S1 = np.cov(X.T)
    S2 = np.cov(Y.T)

    # Optional ridge for numerical stability of S2^{-1}
    if regularize > 0:
        S2 = S2 + regularize * np.eye(p)

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

    p_one = 1 - stats.norm.cdf(Z)
    results = {"stat": Z, "p_value": p_one}

    return results


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
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
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
    r"""
    High-dimensional test of proportional covariance matrices:

        H0: Sigma_x = k * Sigma_y for some k > 0
        H1: Sigma_x is not proportional to Sigma_y

    This implements the trace-based statistic of Tsukuda and Matsuura (2019),
    valid when p can exceed m and n, with m, n on the order of p**delta for
    delta in (1/2, 1). The statistic is

        T = (m*n / (m + n)) * [ a_x2 / a_x1**2 + a_y2 / a_y1**2
                                 - 2 * a_xy / (a_x1 * a_y1) ],

    where
      - a_x1 = tr(Sx)/p
      - a_y1 = tr(Sy)/p
      - a_xy = tr(Sx * Sy)/p
      - a_x2 and a_y2 are unbiased estimates of tr(Sigma_x^2)/p and tr(Sigma_y^2)/p

    The asymptotic variance factor is estimated by

        b_hat^2 = [ m**2/(m**2 + n**2) ] * (a_x2 / a_x1**2)
                + [ n**2/(m**2 + n**2) ] * (a_y2 / a_y1**2).

    We report Z = T / (2 * b_hat), which is approximately N(0, 1) under H0
    (one-sided test rejects for large positive Z). We also report a two-sided
    p-value for convenience.

    Parameters
    ----------
    X : array-like, shape (m, p)
        Sample 1 (rows are observations).
    Y : array-like, shape (n, p)
        Sample 2 (rows are observations).

    Returns
    -------
    dict
        {
          "stat": float,                      # standardized statistic
          "p_value": float,
        }

    Notes
    -----
    - tr(Sigma^2)/p is estimated with the unbiased closed form of
      Srivastava, Yanagihara, and Kubokawa (2014), Eq. (2.5), computed
      from centered data and the scatter matrix V and diagonal matrix D.
    - Since Sx and Sy are independent unbiased estimators of Sigma_x and
      Sigma_y, tr(Sx * Sy) is an unbiased estimator of tr(Sigma_x * Sigma_y).
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X and Y must be 2D arrays (observations × features).")
    m, pX = X.shape
    n, pY = Y.shape
    if pX != pY:
        raise ValueError(
            "X and Y must have the same number of columns (same p)."
        )
    p = pX
    if m < 4 or n < 4:
        raise ValueError(
            "Each group must have at least 4 observations (for unbiased tr(Sigma^2))."
        )

    # Sample covariances with ddof=1
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    Sx = (Xc.T @ Xc) / (m - 1)
    Sy = (Yc.T @ Yc) / (n - 1)

    a_x1 = np.trace(Sx) / p
    a_y1 = np.trace(Sy) / p
    a_xy = np.trace(Sx @ Sy) / p

    a_x2 = tsukuda2019._unbiased_tr_sigma2_per_p(X)
    a_y2 = tsukuda2019._unbiased_tr_sigma2_per_p(Y)

    # Core statistic and variance estimate
    T = (m * n / (m + n)) * (
        a_x2 / (a_x1**2) + a_y2 / (a_y1**2) - 2.0 * a_xy / (a_x1 * a_y1)
    )

    b2_hat = ((m**2) / (m**2 + n**2)) * (a_x2 / (a_x1**2)) + (
        (n**2) / (m**2 + n**2)
    ) * (a_y2 / (a_y1**2))

    b_hat = np.sqrt(b2_hat) if b2_hat > 0 else np.nan
    Z = T / (2.0 * b_hat) if np.isfinite(b_hat) and b_hat > 0 else np.nan

    if single_side:
        p_value = 1.0 - stats.norm.cdf(Z)
    else:
        p_value = 2.0 * stats.norm.sf(abs(Z))

    return {"stat": float(Z), "p_value": float(p_value)}
