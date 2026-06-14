import numpy as np
import scipy.stats as stats  # type: ignore

from . import _hallin2006 as hallin2006
from . import _srivastava_2005 as s2005
from . import _ahmad2015 as ahmad2015
from .utils import (
    covariance_traces,
    result_dict,
    sample_covariance,
    validate_data_matrix,
)


def ahmad2015_sphericity_test(
    X: np.ndarray,
    center: bool = True,
    calibration: str = "auto",
    tail: str = "upper",
):
    """Ahmad (2015) sphericity test.

    Test H0: Sigma = sigma^2 I using T1 = p*E3/E2 - 1.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The input data matrix.
    center : bool, default=True
        If True, center the data by subtracting column means.
    calibration : {"auto", "large_p_small_n", "ratio"}, default="auto"
        Calibration method for computing the standardized z-score.
    tail : {"upper", "two-sided"}, default="upper"
        Whether to calculate upper-tail or two-sided p-value.

    Returns
    -------
    result : dict
        A dictionary containing:
        - 'stat' : float
            The computed test statistic T1.
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

    # Gram matrix (n x n)
    G = X @ X.T

    E1, E2, E3 = ahmad2015._trace_estimators_from_gram(G)
    if E2 <= 0:
        raise ValueError(f"E2 must be positive to form T1; got E2={E2}.")

    T1 = (p * (E3 / E2)) - 1.0

    z, used_cal = ahmad2015._standardize_T(
        T1, n=n, p=p, calibration=calibration
    )

    if tail == "upper":
        pval = float(stats.norm.sf(z))
    elif tail == "two-sided":
        pval = float(2.0 * stats.norm.sf(abs(z)))
    else:
        raise ValueError("tail must be 'upper' or 'two-sided'.")

    return result_dict(float(T1), pval)


def bartlett_sphericity_test(X):
    """Bartlett's test of sphericity.

    Tests the null hypothesis that the correlation matrix equals
    the identity matrix.

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
            P-value for the test.

    References
    ----------
    .. [1] Bartlett, M. S. (1954). "A note on the multiplying
           factors for various chi-square approximations." Journal
           of the Royal Statistical Society, Series B, 16, 296-298.
    """
    X = validate_data_matrix(X)
    n, p = X.shape

    # Compute correlation matrix
    R = np.corrcoef(X, rowvar=False)

    # Check positive definiteness
    sign, logdet = np.linalg.slogdet(R)
    if sign <= 0:
        raise ValueError("Correlation matrix is not positive definite.")

    # Bartlett's test statistic
    stat = -(n - 1 - (2 * p + 5) / 6) * logdet

    # Degrees of freedom
    dof = p * (p - 1) / 2

    # p-value
    p_value = stats.chi2.sf(stat, dof)

    return result_dict(stat, p_value)


def _john_stat(data):
    """
    Compute John's sphericity test statistic.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_features)
        The data matrix, where rows correspond to samples and columns
        to variables.

    Returns
    -------
    U : float
        The value of John’s test statistic.

    Notes
    -----
    The statistic is defined as:

    .. math::

        U = \\frac{\\frac{1}{p}\\,\\mathrm{tr}(S^2)}
            {\\left(\\frac{1}{p}\\,\\mathrm{tr}(S)\\right)^2} - 1

    where :math:`S` is the sample covariance matrix and :math:`p` is
    the number of features.
    """
    _, trace_S, trace_S2 = covariance_traces(data)
    p = data.shape[1]
    U = (1 / p) * trace_S2 / ((1 / p) * trace_S) ** 2 - 1
    return U


# Checked
def john_sphericity(X):
    """John's test for sphericity.

    Tests the null hypothesis that the covariance matrix is
    proportional to the identity matrix (sphericity).

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
    The test statistic is T = U * (np/2), where U is John's
    statistic based on the ratio of trace(S^2) to trace(S)^2.

    References
    ----------
    .. [1] John, S. (1971). "Some optimal multivariate tests."
           Biometrika, 58(1), 123-127.
    """
    X = validate_data_matrix(X)
    n, p = X.shape
    U = _john_stat(X)
    degree_of_freedom = p * (p + 1) / 2 - 1
    stat = U * n * p / 2
    p_value = 1 - stats.chi2.cdf(stat, degree_of_freedom)
    return result_dict(stat, p_value)


# Checked
def srivastava_2005_sphericity(X):
    """Srivastava (2005) test for sphericity.

    High-dimensional test for H0: Sigma = c * I_p for some c > 0.

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


def _spatial_sign_rows(A):
    # A: (m, p) -> row-wise spatial signs
    norms = np.linalg.norm(A, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return A / norms


def _U_tensor(X):
    """
    U[i, j, :] = spatial sign of (X[i] - X[j]).
    Shape: (n, n, p). Diagonal rows are zero.
    """
    X = np.asarray(X)
    n, p = X.shape
    U = np.empty((n, n, p), dtype=float)
    for i in range(n):
        U[i] = _spatial_sign_rows(X[i] - X)  # (n, p)
        U[i, i] = 0.0
    return U


def sk_test(X):
    """High-dimensional Kendall's tau-type sphericity test (SK).

    Implements the leave-one-out U-statistic estimator from Feng & Liu (2017).

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The input data matrix.

    Returns
    -------
    result : dict
        A dictionary containing:
        - 'stat' : float
            The computed test statistic Q.
        - 'p_value' : float
            The computed p-value.
    """
    X = validate_data_matrix(X)
    n, p = X.shape
    if n < 4:
        raise ValueError("Need n >= 4 for the leave-one-out estimator.")

    U = _U_tensor(X)  # (n, n, p)

    # Build S_i = sum_{j != i} U[i, j] U[i, j]^T  (p x p for each i)
    # and W = sum_i S_i
    S_list = []
    W = np.zeros((p, p), dtype=float)
    for i in range(n):
        A_i = U[i]  # (n, p), with A_i[i]=0
        S_i = A_i.T @ A_i  # p x p   (sums over j)
        S_list.append(S_i)
        W += S_i

    sum_tr_Si2 = sum(np.sum(S_i * S_i) for S_i in S_list)
    tr_W2 = float(np.sum(W * W))

    sk_sum_distinct = tr_W2 - 4.0 * sum_tr_Si2 + 2.0 * n * (n - 1)

    denom = n * (n - 1) * (n - 2) * (n - 3)
    tr_hat = sk_sum_distinct / denom

    Q = p * tr_hat - 1.0
    sigma2 = 4.0 * (p - 1) / (n * (n - 1) * (p + 2))  # paper's null variance
    z = Q / np.sqrt(sigma2)
    pval = 1.0 - stats.norm.cdf(z)  # right-tail, as in the paper

    return result_dict(Q, pval)


def muirhead_sphericity_lrt(
    X=None,
    S=None,
    n=None,
    center=True,
    use_bartlett_correction=True,
):
    """Muirhead likelihood ratio test (LRT) for sphericity.

    Tests H0: Sigma = sigma^2 * I_p (sigma^2 unknown) with optional Bartlett correction.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features), optional
        The input data matrix. If provided, S is ignored.
    S : array-like of shape (n_features, n_features), optional
        Sample covariance matrix. If provided, pass n as well.
    n : int, optional
        Sample size used to compute S when S is provided.
    center : bool, default=True
        If X is provided, subtract column means before computing covariance.
    use_bartlett_correction : bool, default=True
        Whether to apply Bartlett small-sample correction.

    Returns
    -------
    result : dict
        A dictionary containing:
        - 'stat' : float
            Chi-square test statistic.
        - 'p_value' : float
            P-value from chi-square distribution.
    """
    if X is None and S is None:
        raise ValueError("Provide either X or S.")
    if S is not None and n is None:
        raise ValueError("If S is provided, also provide n (sample size).")

    if X is not None:
        X = validate_data_matrix(X)
        n, p = X.shape
        Xc = X - np.mean(X, axis=0)
        S_local = (Xc.T @ Xc) / (n - 1)
    else:
        S_local = np.asarray(S, dtype=float)
        if S_local.ndim != 2 or S_local.shape[0] != S_local.shape[1]:
            raise ValueError("S must be a square 2D array.")
        p = S_local.shape[0]

    if n is None:
        n = X.shape[0]

    # Log-determinant (stable) and trace
    sign, logdetS = np.linalg.slogdet(S_local)
    if not np.isfinite(logdetS) or sign <= 0:
        raise ValueError(
            "Sample covariance is not positive definite. "
            "The LRT for sphericity is undefined; consider the Ledoit-Wolf test."
        )
    trS = float(np.trace(S_local))
    t1 = trS / p

    # W and T
    logW = logdetS - p * float(np.log(t1))

    df = (p - 1) * (p + 2) // 2
    if use_bartlett_correction:
        rho = 1.0 - (2.0 * p * p + p + 2.0) / (6.0 * (n - 1.0) * p)
    else:
        rho = 1.0

    T = -(n - 1.0) * rho * logW
    pval = stats.chi2.sf(T, df)

    return result_dict(T, float(pval))


def czz_sphericity_test(X, center=False):
    """Chen-Zhang-Zhong (2010) sphericity test.

    Tests H0: Sigma = sigma^2 * I using location-invariant U-statistics.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The input data matrix.
    center : bool, default=False
        If True, subtract column means before computing the test.

    Returns
    -------
    result : dict
        A dictionary containing:
        - 'stat' : float
            The normal approximation statistic.
        - 'p_value' : float
            The right-tailed p-value.
    """
    X = validate_data_matrix(X)
    n, p = X.shape
    if n < 4:
        raise ValueError("Require n >= 4 so that P4_n is positive.")

    Xc = X - X.mean(axis=0, keepdims=True) if center else X

    # Gram matrix and basic aggregates
    G = Xc @ Xc.T  # shape (n, n)
    d = np.diag(G)  # diagonal
    r = G.sum(axis=1)  # row sums (including diagonal)
    R = r - d  # off-diagonal row sums
    s_off = float(R.sum())  # sum_{i!=j} G_ij
    sumsq_all = float(np.sum(G * G))  # Frobenius norm squared
    sumsq_diag = float(np.sum(d * d))
    sumsq_off = sumsq_all - sumsq_diag
    sum_R2 = float(np.sum(R * R))

    # Falling factorial denominators Pk_n
    P2 = n * (n - 1)
    P3 = P2 * (n - 2)
    P4 = P3 * (n - 3)

    # Y1..Y5
    Y1 = float(np.sum(d)) / n
    Y2 = sumsq_off / P2
    Y3 = s_off / P2
    Y4 = (sum_R2 - sumsq_off) / P3
    Y5 = (s_off * s_off - 4.0 * sum_R2 + 2.0 * sumsq_off) / P4

    # T1, T2
    T1 = Y1 - Y3
    T2 = Y2 - 2.0 * Y4 + Y5
    if T1 <= 0:
        raise ValueError("Nonpositive T1 encountered; check data quality.")

    # CZZ statistic and normal calibration
    U = p * (T2 / (T1 * T1)) - 1.0
    Z = (n * U) / 2.0
    p_norm = float(1.0 - stats.norm.cdf(Z))  # right-tailed

    return result_dict(float(Z), p_norm)


def hallin_rank_sphericity_test(X, method="wilcoxon"):
    """Van der Waerden or Wilcoxon rank-based test for sphericity.

    Tests the null hypothesis of sphericity using spatial ranks.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The input data matrix.
    method : {"wilcoxon", "vdw"}, default="wilcoxon"
        Rank-based score function to use.

    Returns
    -------
    result : dict
        A dictionary containing:
        - 'stat' : float
            The computed test statistic Q.
        - 'p_value' : float
            The computed p-value.
    """
    X = validate_data_matrix(X)
    if method not in {"wilcoxon", "vdw"}:
        raise ValueError("Unrecognized method %s" % method)

    n, k = X.shape
    U, d = hallin2006._center_and_scale(X)
    ranks = stats.rankdata(d, method="average")
    u = ranks / (n + 1)

    if method == "wilcoxon":
        scores = u - 0.5
        score_var = 1 / 12
    else:
        u = np.clip(u, 1e-6, 1 - 1e-6)

        raw_scores = stats.chi.ppf(u, df=k)
        scores = raw_scores - np.mean(raw_scores)
        score_var = np.var(raw_scores, ddof=0)

    Q = hallin2006._compute_statistic(U, scores, k, score_var)
    df = k * (k + 1) // 2 - 1
    pval = 1 - stats.chi2.cdf(Q, df)

    return result_dict(Q, pval)


def _gram_aggregates(X):
    """Return pre-computed aggregates of the n x n Gram matrix G = X X'.

    Parameters
    ----------
    X : array (n, p) - rows are observations (already optionally centred).

    Returns
    -------
    G         : (n, n) Gram matrix
    d         : (n,)  diagonal of G, i.e. squared row-norms
    R         : (n,)  off-diagonal row sums  R_i = sum_{j != i} G_{ij}
    s_off     : float  sum of all off-diagonal entries of G
    sumsq_off : float  sum of squares of all off-diagonal entries of G
    sum_R2    : float  sum_i R_i^2
    sum_d     : float  sum_i d_i = tr(V)
    sum_d_sq  : float  sum_i d_i^2 = tr(D^2)
    sum_dR    : float  sum_i d_i * R_i
    """
    G = X @ X.T
    d = np.diag(G)
    r_full = G.sum(axis=1)
    R = r_full - d  # off-diagonal row sums
    s_off = float(R.sum())
    sumsq_all = float(np.sum(G * G))
    sumsq_diag = float(np.sum(d * d))
    sumsq_off = sumsq_all - sumsq_diag
    sum_R2 = float(np.sum(R * R))
    sum_d = float(d.sum())
    sum_d_sq = float(sumsq_diag)
    sum_dR = float(np.dot(d, R))
    return G, d, R, s_off, sumsq_off, sum_R2, sum_d, sum_d_sq, sum_dR


def _spatial_sign_cov(X):
    """Spatial-sign sample covariance matrix B_hat_n (p x p).

    x_hat_j = sqrt(p) * x_j / ||x_j||  (scaled to have ||x_hat_j|| = sqrt(p))
    B_hat_n = (1/n) * sum_j  x_hat_j @ x_hat_j.T

    Rows of X with zero norm are silently dropped.
    """
    n, p = X.shape
    norms = np.linalg.norm(X, axis=1, keepdims=True)  # (n, 1)
    mask = norms.ravel() > 0
    Xs = X[mask] / norms[mask]  # unit-norm rows
    Xs_scaled = Xs * p**0.5  # scaled to norm sqrt(p)
    B = (Xs_scaled.T @ Xs_scaled) / n  # (p, p)
    return B


def fisher_2010_sphericity_test(X, center=True):
    r"""Fisher-Sun-Gallagher (2010) sphericity test.

    Tests H0 : Sigma = sigma^2 I_p using the ratio of the bias-corrected
    fourth and second arithmetic means of the sample eigenvalues.

    Test statistic (Corollary 1, eq. (5), with scale n/C to give N(0,1))::

        T = n / sqrt(8*(8 + 12*c + c^2)) * (a_hat_4 / a_hat_2^2 - 1)

    where c = p/n, a_hat_2 is Srivastava's (2005) unbiased estimator of
    a_2 = tr(Sigma^2)/p, and a_hat_4 is the unbiased estimator of
    a_4 = tr(Sigma^4)/p (Theorem 1, eq. (2)).

    Under assumptions (A) and (B), T -> N(0,1) as (n,p) -> inf with
    p/n -> c in (0, inf).  The test is right-tailed.

    Parameters
    ----------
    X : array-like (N, p)
        Data matrix; N rows are observations, p columns are variables.
        N must satisfy N >= 8 (n = N-1 >= 7) for tau to be defined.
    center : bool, default True
        If True, subtract column means before computing S.

    Returns
    -------
    dict with keys 'stat' (= T) and 'p_value'.

    References
    ----------
    .. [1] Fisher, T. J., Sun, X., & Gallagher, C. M. (2010). "A new test
           for sphericity of the covariance matrix for high dimensional
           data." Journal of Multivariate Analysis, 101(10), 2554-2570.
    """
    X = validate_data_matrix(X)
    N, p = X.shape
    n = N - 1  # Wishart degrees of freedom

    if n < 7:
        raise ValueError(
            "Fisher (2010) test requires N >= 8 (n = N-1 >= 7) so that "
            "the denominator of tau is non-zero."
        )

    if center:
        X = X - X.mean(axis=0, keepdims=True)

    S = (X.T @ X) / n  # sample covariance, denominator n = N-1

    trS = float(np.trace(S))
    S2 = S @ S
    trS2 = float(np.trace(S2))
    S3 = S2 @ S
    trS3 = float(np.trace(S3))
    trS4 = float(np.trace(S3 @ S))

    # --- Srivastava (2005) unbiased estimator of a_2 = tr(Sigma^2)/p ----
    cn = (n * n) / ((n - 1) * (n + 2))
    a2_hat = cn * (trS2 - trS**2 / n) / p

    # --- Fisher (2010) unbiased estimator of a_4 = tr(Sigma^4)/p --------
    #   a_hat_4 = (tau/p) * [tr(S^4)
    #             + b*tr(S^3)*tr(S) + c**{trS2}^2
    #             + d*tr(S^2)*(trS)^2 + e*(trS)^4]
    denom = n**2 + n + 2
    tau = (n**5 * denom) / (
        (n + 1) * (n + 2) * (n + 4) * (n + 6) * (n - 1) * (n - 2) * (n - 3)
    )
    b_coef = -4.0 / n
    c_star = -(2 * n**2 + 3 * n - 6) / (n * denom)
    d_coef = 2 * (5 * n + 6) / (n * denom)
    e_coef = -(5 * n + 6) / (n**2 * denom)

    a4_hat = (tau / p) * (
        trS4
        + b_coef * trS3 * trS
        + c_star * trS2**2
        + d_coef * trS2 * trS**2
        + e_coef * trS**4
    )

    if a2_hat <= 0:
        raise ValueError(
            f"a_hat_2 = {a2_hat:.6g} <= 0; check data quality or use a "
            "larger sample."
        )

    c_ratio = p / n
    # Scale factor: n / sqrt(8*(8+12c+c^2))  -- see implementation notes
    T = (
        n
        / (8 * (8 + 12 * c_ratio + c_ratio**2)) ** 0.5
        * (a4_hat / a2_hat**2 - 1.0)
    )
    pval = float(stats.norm.sf(T))  # right-tailed
    return result_dict(float(T), pval)


def srivastava_2014_sphericity_test(X):
    r"""Srivastava-Yanagihara-Kubokawa (2014) sphericity test.

    Modification of the Srivastava (2005) test using a new U-statistic
    estimator of a_2 = tr(Sigma^2)/p that is unbiased for *any* distribution
    satisfying mild moment conditions, not only the Gaussian distribution.

    Test statistic (eq. (3.6), scale (n/2) to give N(0,1))::

        T_1 = (n/2) * (a_hat_2 / a_hat_1^2 - 1)

    where n = N - 1 (Wishart df), a_hat_1 = tr(S)/p, and a_hat_2 is the
    new unbiased estimator (eq. (2.5))::

        a_hat_2 = [(N-2)*n*tr(V^2) - N*n*tr(D^2) + (tr V)^2]
                  / [p * N*(N-1)*(N-2)*(N-3)]

    with V = sum_j y_j y_j' (= (N-1)*S, the Wishart scatter matrix),
    D = diag(y_1'y_1, ..., y_N'y_N), y_j = x_j - x_bar.

    This estimator is unbiased for any distribution satisfying (1.1)-(1.3),
    whereas Srivastava (2005)'s a_hat_{2s} is biased under non-normality
    when K_4 != 0.

    Under H0 and as (N, p) -> inf with N = O(p^delta), 1/2 < delta < 1,
    T_1 -> N(0,1).  The test is right-tailed.

    Parameters
    ----------
    X : array-like (N, p)
        Data matrix; N rows are observations, p columns are variables.
        Requires N >= 4.

    Returns
    -------
    dict with keys 'stat' (= T_1) and 'p_value'.

        References
    ----------
    .. [1] Srivastava, M. S., Yanagihara, H., & Kubokawa, T. (2014).
           "Tests for covariance matrices in high dimension with less
           sample size." Journal of Multivariate Analysis, 130, 289-309.
    """
    X = validate_data_matrix(X)
    N, p = X.shape
    n = N - 1  # Wishart df

    if N < 4:
        raise ValueError("Srivastava (2014) test requires N >= 4.")

    # Centre the data; V = Y.T @ Y where Y is the (N, p) centred matrix
    Y = X - X.mean(axis=0, keepdims=True)

    # N x N Gram matrix of centred observations
    G_c = Y @ Y.T
    d_c = np.diag(G_c)  # d_i = ||y_i||^2

    sum_d = float(d_c.sum())  # tr(V)
    sum_d_sq = float(np.dot(d_c, d_c))  # tr(D^2) = sum_i d_i^2

    # tr(V^2) = sum_{i,j} G_c[i,j]^2 = tr(G_c^2)
    trV2 = float(np.sum(G_c * G_c))

    # a_hat_2 from eq. (2.5):
    #   [(N-2)*n * tr(V^2)  -  N*n * tr(D^2)  +  (tr V)^2]
    #   / [p * N*(N-1)*(N-2)*(N-3)]
    # with n = N-1 so (N-2)*n = (N-2)*(N-1) and N*n = N*(N-1).
    num_a2 = (N - 2) * n * trV2 - N * n * sum_d_sq + sum_d**2
    den_a2 = float(p) * N * n * (N - 2) * (N - 3)
    a2_hat = num_a2 / den_a2

    # a_hat_1 = tr(S)/p = tr(V)/(n*p)
    a1_hat = sum_d / (n * p)

    if a1_hat <= 0:
        raise ValueError(
            f"a_hat_1 = {a1_hat:.6g} <= 0; sample covariance is degenerate."
        )

    # Scale (n/2) follows the same convention as the existing
    # srivastava_2005_sphericity implementation -- see implementation notes
    T1 = (n / 2.0) * (a2_hat / a1_hat**2 - 1.0)
    pval = float(stats.norm.sf(T1))  # right-tailed
    return result_dict(float(T1), pval)


def hu_2019_sphericity_test(X, center=True, return_all=False):
    r"""Hu-Li-Liu-Zhou (2019) spatial-sign sphericity tests.

    Tests H0 : Sigma = sigma^2 I_p for elliptical populations using linear
    spectral statistics of the spatial-sign sample covariance matrix::

        B_hat_n = (1/n) sum_j  x_hat_j x_hat_j',   x_hat_j = sqrt(p)*x_j/||x_j||.

    Two test statistics are provided (Theorem 3.1):

    T_1 = alpha_hat_{n,2} - 1
        alpha_hat_{n,2} = beta_hat_{n,2} - c_n,
        beta_hat_{n,j} = p^{-1} tr(B_hat_n^j), c_n = p/n.

    T_2 = alpha_hat_{n,4} - 1  (new; more sensitive to spike alternatives)
        alpha_hat_{n,4} = beta_hat_{n,4} - 4*c_n*beta_hat_{n,3}
                          - 2*c_n*(beta_hat_{n,2})^2
                          + 10*c_n^2*beta_hat_{n,2} - 5*c_n^3.

    Combined max-type statistic (Theorem 3.2)::

        Tm = max(z1, z2)
        z1 = (n*T_1 + 1) / 2
        z2 = (n*T_2 + 6 - c_n) / sqrt(8*(18 + 12*c_n + c_n^2))

    Under H0, n*(T_1, T_2) -> N_2((-1, -6+c), Omega) where Omega has
    omega_11=4, omega_12=24, omega_22=8*(18+12c+c^2).  Tm is right-tailed.

    The combined test Tm is more robust than T_1 or T_2 alone against
    spike-like covariance alternatives.  By default the function returns
    only Tm; set return_all=True to also get the individual tests.
        Parameters
    ----------
    X : array-like (n, p)
        Data matrix; rows are observations.
    center : bool, default True
        If True, subtract column means before computing spatial signs.
    return_all : bool, default False
        If True, return a dict with keys 'T1', 'T2', 'Tm' (each a
        result_dict).  If False, return only the Tm result_dict.

    Returns
    -------
    dict with keys 'stat' and 'p_value' (for Tm), or a dict of three
    result_dicts when return_all=True.

    Notes
    -----
    The combined max test Tm has mild finite-sample size inflation for
    small n or p (see Hu et al. 2019 Table S2).  Use with min(n, p) >= 100
    for well-calibrated type-I error.

    References
    ----------
    .. [1] Hu, J., Li, W., Liu, Z., & Zhou, W. (2019). "High-dimensional
           covariance matrices in elliptical distributions with application
           to spherical test." The Annals of Statistics, 47(1), 527-555.
    """
    X = validate_data_matrix(X)
    n, p = X.shape

    if n < 2:
        raise ValueError("Hu (2019) test requires n >= 2.")

    if center:
        X = X - X.mean(axis=0, keepdims=True)

    c_n = p / n

    # Spatial-sign sample covariance (p x p)
    B = _spatial_sign_cov(X)

    # Spectral moments via eigenvalues (more stable than matrix powers)
    eigs = np.linalg.eigvalsh(B)  # all real, >= 0

    b2 = float(np.sum(eigs**2)) / p
    b3 = float(np.sum(eigs**3)) / p
    b4 = float(np.sum(eigs**4)) / p

    # Bias-corrected spectral moments
    alpha2 = b2 - c_n
    alpha4 = (
        b4
        - 4.0 * c_n * b3
        - 2.0 * c_n * b2**2
        + 10.0 * c_n**2 * b2
        - 5.0 * c_n**3
    )

    T1 = alpha2 - 1.0
    T2 = alpha4 - 1.0

    # Standardised individual statistics
    z1 = (n * T1 + 1.0) / 2.0
    denom_T2 = 8.0 * (18.0 + 12.0 * c_n + c_n**2)
    z2 = (n * T2 + 6.0 - c_n) / denom_T2**0.5

    Tm = max(z1, z2)

    # P-value for Tm: P(Tm > t) = 1 - P(z1 <= t AND z2 <= t)
    # Null correlation: rho = 24 / sqrt(4 * 8*(18+12c+c^2)) = 6/sqrt(2*(18+12c+c^2))
    rho = 6.0 / (2.0 * (18.0 + 12.0 * c_n + c_n**2)) ** 0.5
    cov_mat = np.array([[1.0, rho], [rho, 1.0]])
    bvn_cdf = stats.multivariate_normal.cdf(
        [Tm, Tm], mean=[0.0, 0.0], cov=cov_mat
    )
    pval_Tm = float(1.0 - bvn_cdf)

    if not return_all:
        return result_dict(float(Tm), pval_Tm)

    pval_T1 = float(stats.norm.sf(z1))
    pval_T2 = float(stats.norm.sf(z2))
    return {
        "T1": result_dict(float(z1), pval_T1),
        "T2": result_dict(float(z2), pval_T2),
        "Tm": result_dict(float(Tm), pval_Tm),
    }


def xu_2023_sphericity_test(X, center=False):
    r"""Xu-Zhou-Lin-Feng (2023) adjusted sphericity test for elliptical data.

    Ellipticity-corrected version of the Chen-Zhang-Zhong (2010) sphericity
    test, valid for arbitrary (n,p)-asymptotics under elliptical distributions.

    Adjusted test statistic (p. 257)::

        U_hat_{n,p} = sigma_hat_0^{-1}_{n,p} * p * (p*T_{2,n,p}/T_{3,n,p} - 1)

    where:

    * T_{2,n,p}  - CZZ location-invariant U-statistic estimating tr(Sigma^2).
    * T_{3,n,p}  - 4th-order U-statistic estimating tr^2(Sigma)::

          T_{3,n,p} = Y_tilde_2 - 2*Y_tilde_4 + Y_5

      with Y_tilde_2 = sum_{i1!=i2} d_{i1}*d_{i2} / P2,
           Y_tilde_4 = (sum_d*s_off - 2*sum_dR) / P3,
           d_i = ||x_i||^2, P_k = n*(n-1)*...*(n-k+1).

    * delta_{n,p} - 5th-order U-statistic (kurtosis correction)::

          delta_{n,p} = Y_6 - 4*Y_7 + 2*Y_8 + 4*Y_4 - 3*Y_5

      with Y_6 = sum_i d_i^2 / n,
           Y_7 = sum_i d_i*R_i / P2,  R_i = sum_{j!=i} x_i'x_j,
           Y_8 = Y_tilde_4.

    * sigma_hat_0^2 = 2*n^{-2}*p^2*{3*(p/(p+2))^2*delta^2/T_3^2 - 1}
                    - 4*n^{-2}*p^2*{(p/(p+2))*delta/T_3 - 1}

    Under H0, U_hat -> N(0,1) for any (n,p) with p -> inf as n -> inf,
    under elliptical distributions.  The CZZ test (czz_sphericity_test) has
    inflated type-I error for elliptical data with excess kurtosis kappa>1;
    this test corrects for that.  The test is right-tailed.

    Parameters
    ----------
    X : array-like (n, p)
        Data matrix; n rows are observations, p columns are variables.
        Requires n >= 5.
    center : bool, default False
        The U-statistics are location-invariant without centering.
        Set True to explicitly subtract column means first.

    Returns
    -------
    dict with keys 'stat' (= U_hat) and 'p_value'.

    References
    ----------
    .. [1] Xu, G., Zhou, C., Lin, S., & Feng, Y. (2023). "Adjusted
           covariance matrix U-tests for elliptically distributed data."
           Scandinavian Journal of Statistics, 52, 249-269.
    """
    X = validate_data_matrix(X)
    n, p = X.shape

    if n < 5:
        raise ValueError(
            "Xu (2023) test requires n >= 5 (5th-order U-statistic)."
        )

    if center:
        X = X - X.mean(axis=0, keepdims=True)

    _, d, R, s_off, sumsq_off, sum_R2, sum_d, sum_d_sq, sum_dR = (
        _gram_aggregates(X)
    )

    P2 = n * (n - 1)
    P3 = P2 * (n - 2)
    P4 = P3 * (n - 3)

    # ---- CZZ Y-statistics ------------------------------------------------
    Y2 = sumsq_off / P2
    Y4 = (sum_R2 - sumsq_off) / P3
    Y5 = (s_off**2 - 4.0 * sum_R2 + 2.0 * sumsq_off) / P4

    # CZZ estimators of tr(Sigma) and tr(Sigma^2)
    T1 = sum_d / n - s_off / P2
    T2 = Y2 - 2.0 * Y4 + Y5

    if T1 <= 0:
        raise ValueError(
            "Nonpositive T1 (estimator of tr(Sigma)); check data quality."
        )

    # ---- Y_tilde statistics (use squared norms d_i) ----------------------
    # Y_tilde_2 = sum_{i1 != i2} d_{i1}*d_{i2} / P2
    Ytilde2 = (sum_d**2 - sum_d_sq) / P2

    # Y_tilde_4 = sum_{i1,i2,i3 all distinct} d_{i1}*G_{i2,i3} / P3
    #           = (sum_d * s_off - 2 * sum_dR) / P3
    Ytilde4 = (sum_d * s_off - 2.0 * sum_dR) / P3

    # T3 = Y_tilde_2 - 2*Y_tilde_4 + Y5  (estimator of tr^2(Sigma))
    T3 = Ytilde2 - 2.0 * Ytilde4 + Y5

    if T3 <= 0:
        raise ValueError(
            "Nonpositive T3 (estimator of tr^2(Sigma)); cannot proceed."
        )

    # ---- Y_6, Y_7, Y_8 for delta (5th-order U-statistic) ----------------
    # Y_6 = (1/n) * sum_i d_i^2
    Y6 = sum_d_sq / n

    # Y_7 = sum_{i1 != i2} d_{i1} * G_{i1,i2} / P2  = sum_dR / P2
    Y7 = sum_dR / P2

    # Y_8 = Y_tilde_4 (same quantity, re-used)
    Y8 = Ytilde4

    # delta_{n,p}
    delta = Y6 - 4.0 * Y7 + 2.0 * Y8 + 4.0 * Y4 - 3.0 * Y5

    q = p / (p + 2.0)  # = p/(p+2)
    r = delta / T3  # ratio delta/T3

    sigma_sq_0 = 2.0 / (n**2) * p**2 * (
        3.0 * q**2 * r**2 - 1.0
    ) - 4.0 / (n**2) * p**2 * (q * r - 1.0)

    if sigma_sq_0 <= 0:
        raise ValueError(
            f"sigma_hat_0^2 = {sigma_sq_0:.6g} <= 0; kurtosis correction "
            "is invalid for this dataset."
        )

    sigma_0 = sigma_sq_0**0.5

    U_hat = (p / sigma_0) * (p * T2 / T3 - 1.0)
    pval = float(stats.norm.sf(U_hat))  # right-tailed
    return result_dict(float(U_hat), pval)
