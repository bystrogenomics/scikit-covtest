from typing import Any, Dict, Optional

import numpy as np
import numpy.linalg as la
from scipy.stats import chi2, f, norm  # type: ignore

from . import _ahmad2017 as ahmad2017
from . import _ding as ding2023
from . import _ishii2015 as ishii2015
from . import _tylers as tyler


def ahmad_2015_two_sample(X, Y):
    """
    Perform the Ahmad (2017) location-invariant test for homogeneity
    of high-dimensional covariance matrices.

    Parameters
    ----------
    X : np.ndarray, shape (n1, p)
        First sample.
    Y : np.ndarray, shape (n2, p)
        Second sample.

    Returns
    -------
    result : dict
        Dictionary containing:
        - 'stat': test statistic
        - 'p_value': p-value based on asymptotic normality
    """
    n1, p = X.shape
    n2 = Y.shape[0]

    E1 = ahmad2017.estimate_Ei(X)
    E2 = ahmad2017.estimate_Ei(Y)
    E12 = ahmad2017.estimate_E12(X, Y)

    tau_hat = E1 + E2 - 2 * E12
    a_hat = E12 / p**2
    n0 = 1 / n1 + 1 / n2

    T2 = a_hat * tau_hat / (E12 if E12 != 0 else 1e-10)

    # Asymptotic variance under H0 (Eq. (8) in paper)
    sigma2_T2 = 4 * a_hat**2 * n0**2
    sigma_T2 = np.sqrt(sigma2_T2)

    z_score = T2 / sigma_T2
    p_value = 1 - norm.cdf(z_score)  # upper tail
    return {
        "stat": T2,
        "p_value": p_value,
    }


def boxm_test(x, y, type="chi.squared"):
    """
    Test equality of two covariance matrices via Box's M test.

    This function compares the covariance matrices of two multivariate samples
    with the null hypothesis :math:`H_0: \\Sigma_x = \\Sigma_y`. It computes the
    Box's M log-likelihood ratio statistic using unbiased sample covariances
    (ddof=1) and evaluates a large-sample reference distribution chosen by
    ``type``.

    Parameters
    ----------
    x : array-like of shape (n_samples_x, n_features)
        First data matrix. Rows are samples and columns are features.
    y : array-like of shape (n_samples_y, n_features)
        Second data matrix. Rows are samples and columns are features.
    type : {"chi.squared", "F"}, default="chi.squared"
        Reference distribution used to compute the p-value.

        - "chi.squared": Uses the usual chi-squared approximation with
          degrees of freedom :math:`p(p+1)/2`, with Bartlett-type correction.
        - "F": Uses an F approximation based on higher-order corrections.

    Returns
    -------
    result : dict
        Dictionary with the following keys:

        - ``"stat"`` : float
            The test statistic on the selected reference scale
            (chi-squared or F, depending on ``type``).
        - ``"p_value"`` : float
            The upper-tail p-value under the selected reference distribution.

    Raises
    ------
    ValueError
        If ``x`` and ``y`` do not have the same number of columns
        (features).
    ValueError
        If ``n_features >= n_samples_x`` or ``n_features >= n_samples_y``.
        The test requires invertible sample covariance estimates.
    ValueError
        If ``type`` is not one of {"chi.squared", "F"}.

    Notes
    -----
    - The test assumes independent samples from multivariate normal
      distributions with equal means not required, but equal covariance
      matrices under :math:`H_0`.
    - The sample covariance matrices are computed with ``ddof=1`` and must be
      positive definite. Near-singular matrices can cause numerical issues
      in the log-determinant.
    - For two groups the chi-squared approximation uses
      :math:`\\text{df} = p(p+1)/2`, where :math:`p` is the number of features.
    - Large-sample approximations can be sensitive to strong deviations from
      normality or heavy tails.

    References
    ----------
    .. [1] G. E. P. Box (1949). "A General Distribution Theory for a Class of
           Likelihood Criteria." *Biometrika*, 36(3/4), 317–346.
    .. [2] T. W. Anderson (2003). *An Introduction to Multivariate Statistical
           Analysis*, 3rd ed., Wiley, Sections 8.3–8.4.
    .. [3] A. C. Rencher and W. F. Christensen (2012). *Methods of Multivariate
           Analysis*, 3rd ed., Wiley, Chapter 6.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> Sigma = np.array([[1.0, 0.3, 0.2],
    ...                   [0.3, 1.0, 0.1],
    ...                   [0.2, 0.1, 1.0]])
    >>> X = rng.multivariate_normal(np.zeros(3), Sigma, size=120)
    >>> Y = rng.multivariate_normal(np.zeros(3), Sigma, size=110)
    >>> res = boxm_test(X, Y, type="chi.squared")
    >>> res["p_value"] > 0.05
    True

    >>> # Different covariance in Y should often be detected
    >>> Y_alt = rng.multivariate_normal(np.zeros(3), 1.8 * Sigma, size=110)
    >>> res_alt = boxm_test(X, Y_alt, type="chi.squared")
    >>> res_alt["p_value"] < 0.05
    True
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if x.shape[1] != y.shape[1]:
        raise ValueError("Dimensions do not match")
    if x.shape[1] >= x.shape[0] or y.shape[1] >= y.shape[0]:
        raise ValueError("This is not a high dimensional test")

    n, p = x.shape
    m = y.shape[0]

    # Sample covariance matrices
    s1 = np.cov(x, rowvar=False, bias=False)
    s2 = np.cov(y, rowvar=False, bias=False)

    # Pooled covariance
    s_pooled = ((n - 1) * s1 + (m - 1) * s2) / (n + m - 2)

    # Log-M statistic
    log_M = (
        (n - 1) * np.log(la.det(s1))
        + (m - 1) * np.log(la.det(s2))
        - (n + m - 2) * np.log(la.det(s_pooled))
    ) / 2.0

    if type == "chi.squared":
        c1 = (
            (1 / (n - 1) + 1 / (m - 1) - 1 / (n + m - 2))
            * (2 * p**2 + 3 * p - 1)
            / (6 * (p + 1))
        )
        test_statistic = -2 * (1 - c1) * log_M
        df = p * (p + 1) / 2
        p_value = chi2.sf(test_statistic, df=df)
    elif type == "F":
        c1 = (
            (1 / (n - 1) + 1 / (m - 1) - 1 / (n + m - 2))
            * (2 * p**2 + 3 * p - 1)
            / (6 * (p + 1))
        )
        c2 = (
            (p - 1)
            * (p + 2)
            / 6
            * (1 / (n - 1) ** 2 + 1 / (m - 1) ** 2 - 1 / (n + m - 2) ** 2)
        )
        a1 = p * (p + 1) / 2
        a2 = (a1 + 2) / abs(c2 - c1**2)
        b1 = (1 - c1 - a1 / a2) / a1
        b2 = (1 - c1 - 2 / a2) / a2

        if c2 > c1**2:
            test_statistic = -2 * b1 * log_M
        else:
            test_statistic = -(a2 * b2 * log_M) / (a1 * (1 + 2 * b2 * log_M))

        p_value = f.sf(test_statistic, dfn=a1, dfd=a2)
    else:
        raise ValueError("type must be either 'chi.squared' or 'F'")

    return {"stat": test_statistic, "p_value": p_value}


def ishii_two_sample(X1, X2, test="full"):
    """
    Perform equality test of two covariance matrices using NR PCA.

    Parameters
    ----------
    X1 : np.ndarray, shape (d, n1)
        First data matrix.
    X2 : np.ndarray, shape (d, n2)
        Second data matrix.
    test: str, options = {'full','leading','direction'}
        Type of test

    Returns
    -------
    result : dict
    """
    if X1.shape[1] < 1000:
        raise Warning(
            "Ishii et al (2015) known to be unreliable when d is small"
        )
    d, n1 = X1.shape
    _, n2 = X2.shape
    nu1 = n1 - 1
    nu2 = n2 - 1

    l1, h1, _ = ishii2015.noise_reduction_pca(X1, rank=1)
    l2, h2, _ = ishii2015.noise_reduction_pca(X2, rank=1)
    lambda1 = l1[0]
    lambda2 = l2[0]

    kappa1 = (
        np.trace(
            (X1 - X1.mean(1, keepdims=True)).T
            @ (X1 - X1.mean(1, keepdims=True))
            / nu1
        )
        - lambda1
    )
    kappa2 = (
        np.trace(
            (X2 - X2.mean(1, keepdims=True)).T
            @ (X2 - X2.mean(1, keepdims=True))
            / nu2
        )
        - lambda2
    )
    gamma_hat = max(kappa1 / kappa2, kappa2 / kappa1)
    h_dot = np.abs(h1[:, 0] @ h2[:, 0])
    h_star = (
        (h_dot + 1 / h_dot) / 2
        if lambda1 >= lambda2
        else (h_dot + 1 / h_dot) / 2
    )

    F1 = lambda1 / lambda2
    F2 = F1 * (h_star if lambda1 >= lambda2 else 1 / h_star)
    F3 = F2 * (gamma_hat if lambda1 >= lambda2 else 1 / gamma_hat)

    if test == "full":
        stat = F3
        p_value = 2 * min(f.cdf(F3, nu1, nu2), 1 - f.cdf(F3, nu1, nu2))
    elif test == "leading":
        stat = F1
        p_value = 2 * min(f.cdf(F1, nu1, nu2), 1 - f.cdf(F1, nu1, nu2))
    elif test == "direction":
        stat = F2
        p_value = 2 * min(f.cdf(F2, nu1, nu2), 1 - f.cdf(F2, nu1, nu2))
    else:
        raise ValueError("Unrecognized option %s" % test)

    return {"stat": stat, "p_value": p_value}


def _schott_2001_two_sample_stat(matrix_list):
    len_groups = len(matrix_list)
    samplecov = []
    ns = []
    ntot = 0.0
    p = matrix_list[0].shape[1]
    Apool = np.zeros((p, p))

    for mats in matrix_list:
        ni = mats.shape[0]
        covar = np.cov(mats, rowvar=False, bias=False)
        samplecov.append(covar)
        ns.append(ni)
        ntot += ni - 1
        Apool += covar * (ni - 1)

    ns = np.array(ns)
    pooledCov = Apool / ntot
    invPooled = np.linalg.inv(pooledCov)

    singlesum = 0.0
    doublesum = 0.0

    for i in range(len_groups):
        ni = ns[i] - 1
        Si = samplecov[i]
        term_i = invPooled @ Si @ invPooled
        singlesum += ni / ntot * np.trace(Si @ term_i)

        for j in range(len_groups):
            nj = ns[j] - 1
            Sj = samplecov[j]
            term_j = invPooled @ Sj @ invPooled
            doublesum += ni * nj / (ntot**2) * np.trace(Si @ term_j)

    stat = ntot / 2.0 * (singlesum - doublesum)
    return stat


def schott_2001(X, Y):
    """
    Schott (2001) homogeneity of covariance matrices test.
    Uses asymptotic normal distribution under H0.
    """
    # k = len(x)
    k = 2
    p = X.shape[1]
    matrix_ls = [X, Y]
    stat = _schott_2001_two_sample_stat(matrix_ls)

    # Asymptotic mean and variance under H0
    mu = 0.5 * p * (p + 1) * (k - 1)
    sigma2 = p * (p + 1) * (k - 1)

    zstat = (stat - mu) / np.sqrt(sigma2)
    p_value = 2 * (1 - norm.cdf(abs(zstat)))

    return {
        "stat": zstat,
        "p_value": p_value,
    }


def _srivastava_yanagihara_stat(x):
    len_x = len(x)
    pmat = x[1]
    p = pmat.shape[1]
    ntot = 0
    ns = np.zeros(len_x)
    a2i = np.zeros(len_x)
    a1i = np.zeros(len_x)
    samplecov = []
    Apool = np.zeros((p, p))

    for i in range(len_x):
        mats = x[i]
        n = mats.shape[0]
        ns[i] = n
        covar = np.cov(mats, rowvar=False)
        samplecov.append(covar)

        covartrace = np.trace(covar)
        covar2trace = np.trace(covar @ covar)
        a2i[i] = ((n - 1) ** 2 / (p * (n - 2) * (n + 1))) * (
            covar2trace - (1.0 / (n - 1)) * covartrace**2
        )
        a1i[i] = covartrace / p

        ntot += n - 1
        Apool += covar * (n - 1)

    pooledCov = Apool / ntot
    pooledcov2trace = np.trace(pooledCov @ pooledCov)
    pooledcovtrace = np.trace(pooledCov)
    a2 = (ntot**2 / (p * (ntot - 1) * (ntot + 2))) * (
        pooledcov2trace - (1.0 / ntot) * pooledcovtrace**2
    )
    a1 = pooledcovtrace / p

    a3 = (1.0 / (ntot * (ntot**2 + 3 * ntot + 4))) * (
        np.trace(Apool @ Apool @ Apool) / p
        - 3.0 * ntot * (ntot + 1) * p * a2 * a1
        - ntot * p**2 * a1**3
    )

    c0 = ntot**4 + 6 * ntot**3 + 21 * ntot**2 + 18 * ntot
    c1 = 4 * ntot**3 + 12 * ntot**2 + 18 * ntot
    c2 = 6 * ntot**2 + 4 * ntot
    c3 = 2 * ntot**3 + 5 * ntot**2 + 7 * ntot

    a4 = (1.0 / c0) * (
        np.trace(Apool @ Apool @ Apool @ Apool) / p
        - c1 * a1
        - c2 * p * a1**2 * a2
        - c3 * a2**2
        - ntot * p**3 * a1**4
    )

    ksi2i = np.zeros(len_x)
    gammai = np.zeros(len_x)
    gammabarnum = 0
    gammabardem = 0

    for i in range(len_x):
        ksi2i[i] = (
            4.0
            / (ns[i] - 1) ** 2
            * (
                (a2**2 / a1**4)
                + 2.0
                * (ns[i] - 1)
                / p
                * ((a2**3 / a1**6) - 2.0 * a2 * a3 / a1**5 + a4 / a1**4)
            )
        )

        gammai[i] = a2i[i] / a1i[i] ** 2
        gammabarnum += gammai[i] / ksi2i[i]
        gammabardem += 1.0 / ksi2i[i]

    gammabar = gammabarnum / gammabardem

    stat = 0
    for i in range(len_x):
        stat += (gammai[i] - gammabar) ** 2 / ksi2i[i]

    return stat


def srivastava_yanagihara_two_sample(X, Y):
    matrix_ls = [X, Y]

    # Compute the statistic
    statistic = _srivastava_yanagihara_stat(matrix_ls)
    parameter = len(matrix_ls) - 1

    p_value = 1 - chi2.cdf(statistic, parameter)

    results = {
        "stat": statistic,
        "p_value": p_value,
    }
    return results


def _srivastava_2007_stat(x):
    len_x = len(x)
    pmat = x[1]
    p = pmat.shape[1]
    ntot = 0
    Apool = np.zeros((p, p))
    ns = np.zeros(len_x)
    a2i = np.zeros(len_x)
    samplecov = []

    for i in range(len_x):
        mats = x[i]
        n = mats.shape[0]
        ns[i] = n
        covar = np.cov(mats, rowvar=False)
        samplecov.append(covar)

        covar2trace = np.trace(covar @ covar)
        covartrace = np.trace(covar)
        a2i[i] = (
            (n - 1) ** 2
            / (p * (n - 2) * (n + 1))
            * (covar2trace - (1.0 / (n - 1)) * covartrace**2)
        )

        ntot += n - 1
        Apool += covar * (n - 1)

    pooledCov = Apool / ntot
    pooledcov2trace = np.trace(pooledCov @ pooledCov)
    pooledcovtrace = np.trace(pooledCov)
    a2 = (
        ntot**2
        / (p * (ntot - 1) * (ntot + 2))
        * (pooledcov2trace - (1.0 / ntot) * pooledcovtrace**2)
    )

    a1 = pooledcovtrace / p

    c0 = ntot**4 + 6 * ntot**3 + 21 * ntot**2 + 18 * ntot
    c1 = 4 * ntot**3 + 12 * ntot**2 + 18 * ntot
    c2 = 6 * ntot**2 + 4 * ntot
    c3 = 2 * ntot**3 + 5 * ntot**2 + 7 * ntot

    a4 = (1.0 / c0) * (
        np.trace(Apool @ Apool @ Apool @ Apool) / p
        - c1 * a1
        - c2 * a1**2 * a2
        - c3 * a2**2
        - ntot * a1**4 * p**3
    )

    eta2i = np.zeros(len_x)
    abarnum = 0
    abardem = 0

    for i in range(len_x):
        eta2i[i] = (
            4.0
            / (ns[i] - 1) ** 2
            * a2**2
            * (1.0 + 2.0 * (ns[i] - 1) * a4 / (p * a2**2))
        )

        abarnum += a2i[i] / eta2i[i]
        abardem += 1.0 / eta2i[i]

    abar = abarnum / abardem

    stat = 0
    for i in range(len_x):
        stat += (a2i[i] - abar) ** 2 / eta2i[i]

    return stat


def srivastava_two_sample_2007(X, Y):
    matrix_ls = [X, Y]

    # Compute the statistic
    statistic = _srivastava_2007_stat(matrix_ls)

    parameter = len(matrix_ls) - 1

    p_value = 1 - chi2.cdf(statistic, parameter)

    results = {
        "stat": statistic,
        "p_value": p_value,
    }
    return results


def wald_two_sample(x, y):
    """
    Two-sample Wald test for equality of covariance matrices.

    Reference:
        J. R. Schott (2007).
        "A test for the equality of covariance matrices when the dimension
        is large relative to the sample sizes."
        Computational Statistics and Data Analysis, 51(12):6535–6542.

    Parameters
    ----------
    x : array-like, shape (n, p)
        Data matrix for group 1 with rows as samples and columns as
        variables.
    y : array-like, shape (m, p)
        Data matrix for group 2 with rows as samples and columns as
        variables.

    Returns
    -------
    result : dict
        Dictionary containing:
        - "test_statistic": the Wald test statistic
        - "p_value": the corresponding chi-squared p-value
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if x.shape[1] != y.shape[1]:
        raise ValueError("Dimensions do not match")
    if x.shape[1] >= x.shape[0] or y.shape[1] >= y.shape[0]:
        raise ValueError("This is not a high dimensional test")

    n, p = x.shape
    m = y.shape[0]

    # Sample covariances
    s1 = np.cov(x, rowvar=False, bias=False)
    s2 = np.cov(y, rowvar=False, bias=False)

    # Pooled covariance
    s_pooled = ((n - 1) * s1 + (m - 1) * s2) / (n + m - 2)

    # Inverse pooled covariance
    s_pooled_inv = la.inv(s_pooled)

    # Quadratic forms
    term1 = (
        (n - 1) / (n + m - 2) - (n - 1) ** 2 / (n + m - 2) ** 2
    ) * np.trace((s1 @ s_pooled_inv) @ (s1 @ s_pooled_inv))
    term2 = (
        (m - 1) / (n + m - 2) - (m - 1) ** 2 / (n + m - 2) ** 2
    ) * np.trace((s2 @ s_pooled_inv) @ (s2 @ s_pooled_inv))
    term3 = (
        -2
        * (n - 1)
        * (m - 1)
        / (n + m - 2) ** 2
        * np.trace((s1 @ s_pooled_inv) @ (s2 @ s_pooled_inv))
    )

    # Wald statistic
    test_statistic = (n + m - 2) / 2 * (term1 + term2 + term3)

    # Chi-squared p-value
    df = p * (p + 1) / 2
    p_value = chi2.sf(test_statistic, df=df)

    return {"stat": test_statistic, "p_value": p_value}


def tyler_two_sample(X1, X2, unknown_mean=False):
    n1, p = X1.shape
    n2, _ = X2.shape

    # Tyler shapes (yours are fine; just ensure centering consistent with branch)
    if unknown_mean:
        mu1 = tyler.robust_location(
            X1
        )  # weights ∝ ||x - xbar||^{-2}, as in paper
        mu2 = tyler.robust_location(X2)
        X1c, X2c = X1 - mu1, X2 - mu2
        M1 = tyler.tylers_M(X1c)
        M2 = tyler.tylers_M(X2c)
        c1, c2 = p / (n1 - 1), p / (n2 - 1)
    else:
        M1 = tyler.tylers_M(X1)
        M2 = tyler.tylers_M(X2)
        c1, c2 = p / n1, p / n2

    A = la.solve(M1, M2)
    trA = np.trace(A)
    trA2 = np.trace(A @ A)
    D_hat = p * (trA2 / (trA**2)) - 1.0

    if unknown_mean:
        T2_tr = D_hat - (n1 - 1) / (n1 - p - 1) - (p / (n2 - 1))
        mu, sigma2 = tyler._mu_sigma2(c1, c2)
    else:
        T2_tr = D_hat - (c1 / (1 - c1)) - c2
        mu, sigma2 = tyler._mu_sigma2(c1, c2)

    z = (p * T2_tr - mu) / np.sqrt(sigma2)
    return {"stat": z, "p_value": 1 - norm.cdf(z)}


# Checked
def cai_2013_two_sample(X: np.ndarray, Y: np.ndarray) -> dict:
    """
    Perform the CLX2013 test.

    Parameters
    ----------
    X : np.ndarray
        First matrix.
    Y : np.ndarray
        Second matrix.

    Returns
    -------
    test_result : dict
        Test statistic and p-value.
    """
    n1, p = X.shape
    n2 = Y.shape[0]

    W1 = X - np.mean(X, axis=0)
    W2 = Y - np.mean(Y, axis=0)

    S1 = np.dot(W1.T, W1) / n1
    S2 = np.dot(W2.T, W2) / n2

    Theta1 = np.zeros((p, p))
    Theta2 = np.zeros((p, p))

    for i in range(n1):
        Theta1 += (1 / n1) * (np.outer(W1[i, :], W1[i, :]) - S1) ** 2
    for i in range(n2):
        Theta2 += (1 / n2) * (np.outer(W2[i, :], W2[i, :]) - S2) ** 2

    W = (S1 - S2) / np.sqrt(Theta1 / n1 + Theta2 / n2)
    M = W**2
    M_n = np.max(M)

    TSvalue = M_n - 4 * np.log(p) + np.log(np.log(p))
    pvalue = 1 - np.exp(-1 / np.sqrt(8 * np.pi) * np.exp(-TSvalue / 2))

    return {"stat": TSvalue, "p_value": pvalue}


def he_2018_two_sample(
    X: np.ndarray, Y: np.ndarray, N: Optional[int] = None, alpha: float = 0.05
) -> dict:
    """
    High-dimensional two-sample covariance matrix testing via super-diagonals

    Parameters
    ----------
    X : np.ndarray
        First matrix.
    Y : np.ndarray
        Second matrix.
    N : Optional[int], optional
        Parameter for the test, by default None.
    alpha : float, optional
        Significance level, by default 0.05.

    Returns
    -------
    test_result : dict
        Test result.
    """
    if N is None:
        N = int(np.floor(X.shape[1] ** 0.7))

    def double_sum(X1, X2):
        result = np.sum(X1, axis=0) * np.sum(X2, axis=0) - np.sum(
            X1 * X2, axis=0
        )
        return result

    def triple_sum(X1, X2, X3):
        result = (
            double_sum(X1, X2) * np.sum(X3, axis=0)
            - double_sum(X1 * X3, X2)
            - double_sum(X1, X2 * X3)
        )
        return result

    def quad_sum(X1, X2, X3, X4):
        result = (
            triple_sum(X1, X2, X3) * np.sum(X4, axis=0)
            - triple_sum(X1 * X4, X2, X3)
            - triple_sum(X1, X2 * X4, X3)
            - triple_sum(X1, X2, X3 * X4)
        )
        return result

    def di(X, q):
        n, p = X.shape
        X1 = X[:, : p - q]
        X2 = X[:, q:]
        D_1 = np.sum(double_sum(X1 * X2, X1 * X2))
        D_2 = np.sum(triple_sum(X1, X2, X1 * X2))
        D_3 = np.sum(quad_sum(X1, X2, X1, X2))
        result = (
            1 / (n * (n - 1)) * D_1
            - 2 / (n * (n - 1) * (n - 1)) * D_2
            + 1 / (n * (n - 1) * (n - 2) * (n - 3)) * D_3
        )
        return result

    def dc(X1, X2, q):
        n1, p = X1.shape
        n2 = X2.shape[0]
        X11 = X1[:, : p - q]
        X12 = X1[:, q:]
        X21 = X2[:, : p - q]
        X22 = X2[:, q:]
        Dc_1 = np.sum(np.sum(X11 * X12, axis=0) * np.sum(X21 * X22, axis=0))
        Dc_2 = np.sum(double_sum(X11, X12) * np.sum(X21 * X22, axis=0))
        Dc_3 = np.sum(np.sum(X11 * X12, axis=0) * double_sum(X21, X22))
        Dc_4 = np.sum(double_sum(X11, X12) * double_sum(X21, X22))
        result = (
            Dc_1 / (n1 * n2)
            - Dc_2 / (n1 * (n1 - 1) * n2)
            - Dc_3 / (n1 * n2 * (n2 - 1))
            + Dc_4 / (n1 * (n1 - 1) * n2 * (n2 - 1))
        )
        return result

    def sq(X1, X2, q):
        result = di(X1, q) + di(X2, q) - 2 * dc(X1, X2, q)
        return result

    def ri(X, q):
        n, p = X.shape
        X = X - np.mean(X, axis=0)
        X1 = X[:, : p - q]
        X2 = X[:, q:]
        Y = X1 * X2
        Y = Y - np.sum(Y, axis=0) / (n - 1)
        YYt2 = np.dot(Y, Y.T) ** 2
        result = (np.sum(YYt2) - np.sum(np.diag(YYt2))) / (n * (n - 1))
        return result

    def rc(X1, X2, q):
        n1, p = X1.shape
        X1 = X1 - np.mean(X1, axis=0)
        X11 = X1[:, : p - q]
        X12 = X1[:, q:]
        Y1 = X11 * X12
        Y1 = Y1 - np.sum(Y1, axis=0) / (n1 - 1)

        n2 = X2.shape[0]
        X2 = X2 - np.mean(X2, axis=0)
        X21 = X2[:, : p - q]
        X22 = X2[:, q:]
        Y2 = X21 * X22
        Y2 = Y2 - np.sum(Y2, axis=0) / (n2 - 1)
        result = np.sum((np.dot(Y1, Y2.T)) ** 2) / (n1 * n2)

        return result

    def v2(X1, X2, q):
        n1 = X1.shape[0]
        n2 = X2.shape[0]
        result = (
            ri(X1, q) * 2 / (n1 * (n1 - 1))
            + ri(X2, q) * 2 / (n2 * (n2 - 1))
            + rc(X1, X2, q) * 4 / (n1 * n2)
        )
        return result

    def one_super(X1, X2, q):
        chi = sq(X1, X2, q) ** 2 / v2(X1, X2, q)
        result = chi2.sf(chi, 1)
        return result

    p_values = [one_super(X, Y, i) for i in range(N + 1)]

    return {"stat": 0, "p_value": p_values}


def chang2016(
    X: np.ndarray,
    Y: np.ndarray,
    J: int = 2500,
    seed: int = 2021,
) -> Dict[str, Any]:
    """
    Perform the Two-Sample HD test for the equality of two covariance matrices from
    'Chang, J., Zhou, W., Zhou, W.-X., and Wang, L. (2016). Comparing large covariance matrices
    under weak conditions on the dependence structure and its application to gene clustering.'

    Parameters
    ----------
    X : np.ndarray
        The n1 x p data matrix for sample 1.
    Y : np.ndarray
        The n2 x p data matrix for sample 2.
    J : int, optional
        The number of permutations, by default 2500.
    seed : int, optional
        The random seed for reproducibility, by default 2021.
    dname : str, optional
        The name of the data, by default "X and Y".

    Returns
    -------
    hd_res : dict
        A dictionary containing the test statistics, p-value,
        alternative hypothesis, method, and data names.
    """
    X = np.asarray(X)
    Y = np.asarray(Y)

    n1, p = X.shape
    n2 = Y.shape[0]

    if Y.shape[1] != p:
        raise ValueError("Different dimensions of X and Y.")

    scalev = np.tile(np.concatenate([np.ones(n1) / n1, np.ones(n2) / n2]), J)
    Sx = np.cov(X, rowvar=False) * (n1 - 1) / n1
    Sy = np.cov(Y, rowvar=False) * (n2 - 1) / n2

    xa = X - np.mean(X, axis=0)
    ya = Y - np.mean(Y, axis=0)

    vx = ((xa**2).T @ (xa**2)) / n1 - 2 / n1 * ((xa.T @ xa) * Sx) + Sx**2
    vy = ((ya**2).T @ (ya**2)) / n2 - 2 / n2 * ((ya.T @ ya) * Sy) + Sy**2

    with np.errstate(invalid="ignore"):
        deno = np.sqrt(vx / n1 + vy / n2)
    numo = np.abs(Sx - Sy)
    Tnm = np.max(numo / deno)

    xat = xa.T / n1
    yat = ya.T / n2
    ts = np.zeros(J)

    rng = np.random.default_rng(seed)
    for j in range(J):
        g = rng.standard_normal(n1 + n2)
        scalev = np.concatenate([np.ones(n1) / n1, np.ones(n2) / n2])
        g *= scalev
        atmp = np.sum(g[:n1])
        btmp = np.sum(g[n1:])

        ts1 = ((xa * g[:n1][:, np.newaxis]) - (xat.T * atmp)).T @ xa
        ts2 = ((ya * g[n1:][:, np.newaxis]) - (yat.T * btmp)).T @ ya

        ts[j] = np.max(np.abs(ts1 - ts2) / deno)

    hd_res = {
        "stat": Tnm,
        "p_value": np.mean(ts >= Tnm),
    }
    return hd_res


def schott2007(X: np.ndarray, Y: np.ndarray) -> Dict[str, Any]:
    """
    Perform the Two-Sample Scott test for the equality of two covariance matrices from
    'Schott, J. R. (2007). A test for the equality of covariance matrices
    when the dimension is large relative to the sample size.'

    Parameters
    ----------
    X : np.ndarray
        The n1 x p data matrix for sample 1.
    Y : np.ndarray
        The n2 x p data matrix for sample 2.
    dname : str, optional
        The name of the data, by default "X and Y".

    Returns
    -------
    sc_res : dict
        A dictionary containing the test statistics, p-value,
        alternative hypothesis, method, and data names.
    """
    X = np.asarray(X)
    Y = np.asarray(Y)

    p = X.shape[1]
    n1 = X.shape[0]
    n2 = Y.shape[0]

    if Y.shape[1] != p:
        raise ValueError("Different dimensions of X and Y.")

    Sx = np.cov(X, rowvar=False) * (n1 - 1) / n1
    Sy = np.cov(Y, rowvar=False) * (n2 - 1) / n2

    Sxx = Sx * n1 / (n1 - 1)
    Syy = Sy * n2 / (n2 - 1)

    SsS = (Sxx * n1 + Syy * n2) / (n1 + n2)

    eta1 = ((n1 - 1) + 2) * ((n1 - 1) - 1)
    eta2 = ((n2 - 1) + 2) * ((n2 - 1) - 1)
    d1 = (1 - (n1 - 1 - 2) / eta1) * np.sum(np.diag(Sxx @ Sxx))
    d2 = (1 - (n2 - 1 - 2) / eta2) * np.sum(np.diag(Syy @ Syy))
    d3 = 2 * np.sum(np.diag(Sxx @ Syy))
    d4 = (n1 - 1) / eta1 * np.sum(np.diag(Sxx)) ** 2
    d5 = (n2 - 1) / eta2 * np.sum(np.diag(Syy)) ** 2
    th = (
        4
        * (((n1 + n2 - 2) / ((n1 - 1) * (n2 - 1))) ** 2)
        * (
            (n1 + n2 - 2) ** 2
            / ((n1 + n2) * (n1 + n2 - 2 - 1))
            * (
                np.sum(np.diag(SsS @ SsS))
                - (np.sum(np.diag(SsS))) ** 2 / (n1 + n2 - 2)
            )
        )
        ** 2
    )
    Sc = (d1 + d2 - d3 - d4 - d5) / np.sqrt(th)

    sc_p = (1 - norm.cdf(np.abs(Sc))) * 2
    sc_res = {"stat": np.abs(Sc), "p_value": sc_p}

    return sc_res


def ding2023_two_sample(
    X: np.ndarray,
    Y: np.ndarray,
    n: Optional[int] = None,
    k: int = 100,
    const: Optional[float] = None,
    alpha: float = 0.05,
    epsilon: float = 0.05,
    thres: Optional[float] = None,
    calib: bool = False,
    seed: int = 2021,
) -> dict:
    """
    Perform a two-sample test multiple times.

    Parameters
    ----------
    X : np.ndarray
        First matrix.
    Y : np.ndarray
        Second matrix.
    n : Optional[int], optional
        Sample size, by default None.
    k : int, optional
        Number of iterations, by default 100.
    const : Optional[float], optional
        Constant value, by default None.
    alpha : float, optional
        Significance level, by default 0.05.
    epsilon : float, optional
        Tolerance value, by default 0.05.
    thres : Optional[float], optional
        Threshold value, by default None.
    calib : bool, optional
        Calibration flag, by default False.
    seed : int, optional
        Random seed, by default 2021.

    Returns
    -------
    test_result : dict
        Result of the two-sample test.
    """
    if const is not None and (
        not isinstance(const, float) or not (0.1 <= const <= 10)
    ):
        raise ValueError("const must be a float between 0.1 and 10.")
    if not isinstance(alpha, float) or not (0 < alpha < 1):
        raise ValueError("alpha must be a float between 0 and 1.")
    if not isinstance(epsilon, float) or not (0 < epsilon < 1):
        raise ValueError("epsilon must be a float between 0 and 1.")
    if thres is not None and (not isinstance(thres, float) or thres <= 0):
        raise ValueError("thres must be a float greater than 0.")

    rng = np.random.default_rng(seed)
    reject, df, statistic = 0, 0, 0

    if n is None:
        n1, n2 = X.shape[0], Y.shape[0]
        n = int(min(max(n1, n2) / 2, n1, n2)) - 5

    if thres is None:
        if not calib:
            thres = 2.6
        else:
            thres = ding2023.calibration(
                n1=X.shape[0],
                n2=Y.shape[0],
                p=X.shape[1],
                n=n,
                alpha=alpha,
                const=0.5,
                iterations=100,
                seed=seed,
            )

    if const is None:
        const = ding2023.c_tuning(
            X, Y, n, alpha=alpha, epsilon=epsilon, thres=thres, seed=seed
        )["c"]

    for _ in range(k):
        result = ding2023.two_sample_test_(
            X,
            Y,
            n,
            alpha=alpha,
            const=const,
            epsilon=epsilon,
            thres=thres,
            seed=rng.integers(int(1e6)),
        )
        if isinstance(result, dict) and result["efficient"]:
            df += 1
            statistic += result["statistic"] ** 2
            if result["c"] == 1:
                reject += 1

    if df < 10:
        pvalue = 0
        statistic = 10000
    else:
        pvalue = 1 - chi2.cdf(statistic, df)

    return {
        "stat": statistic,
        "p_value": pvalue,
    }
