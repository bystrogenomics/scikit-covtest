import numpy as np
import scipy.stats as stats  # type: ignore

from . import _hallin2006 as hallin2006
from . import _srivastava_2005 as s2005
from .utils import validate_data_matrix
from . import _ahmad2015 as ahmad2015


def ahmad2015_sphericity_nonormality(
    X: np.ndarray, center: bool = True
) -> dict:
    """
    Ahmad & von Rosen sphericity test.

    Tests:
      H0: Sigma = sigma^2 I

    Statistic:
      T1 = p*(E3/E2) - 1

    Null calibration:
      z = sqrt(n)*T1/2 ~ N(0,1) (asymptotic)
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be 2D (n, p).")
    n, p = X.shape
    if n < 3:
        raise ValueError("Need n >= 3.")

    if center:
        X = ahmad2015._center_columns(X)

    G = X @ X.T
    diag = np.diag(G)
    sum_diag = float(diag.sum())
    sum_diag2 = float(np.dot(diag, diag))

    E2 = ((sum_diag * sum_diag) - sum_diag2) / (n * (n - 1))
    fro2 = float(np.sum(G * G))
    E3 = (fro2 - sum_diag2) / (n * (n - 1))

    if not (np.isfinite(E2) and np.isfinite(E3)):
        raise FloatingPointError("Non-finite E2/E3 encountered.")
    if E2 <= 0:
        raise ValueError(
            "E2 is non-positive; T1 is not reliable (degenerate data)."
        )

    T1 = p * (E3 / E2) - 1.0
    z = (np.sqrt(n) * T1) / 2.0
    pvalue = 2.0 * (1.0 - norm.cdf(abs(z)))

    return {
        "stat": float(z),
        "pvalue": float(pvalue),
    }


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

    return {"stat": stat, "p_value": p_value}


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
    n, p = data.shape
    sample_cov_matrix = np.cov(data, rowvar=False)
    trace_S = np.trace(sample_cov_matrix)
    trace_S2 = np.trace(np.dot(sample_cov_matrix, sample_cov_matrix))
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
    results = {"stat": stat, "p_value": p_value}
    return results


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
    S = np.cov(X.T)
    T_1 = s2005.T_1_stat(S, n)
    z_stat = (n / 2) * T_1
    p_value = 1 - stats.norm.cdf(z_stat)
    results = {"stat": z_stat, "p_value": p_value}
    return results


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
    """
    High-dimensional Kendall's tau–type sphericity test (SK), exact leave-one-out U-statistic.
    Implements the estimator in Feng & Liu (2017) with an O(n^2 p^2) reduction.

    Returns:
        dict(statistic=Q, z=z, p_value=p)
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

    return {"stat": Q, "p_value": pval}


def muirhead_sphericity_lrt(
    X=None,
    S=None,
    n=None,
    center=True,
    use_bartlett_correction=True,
):
    """
    Muirhead LRT for H0: Sigma = sigma^2 * I_p (sigma^2 unknown), with Bartlett correction.

    Statistic and calibration
    -------------------------
    W = |S| / ( (tr(S)/p)^p )
    T = -(n - 1) * rho * log(W)  approx  chi2_df  under H0

    where:
      - S is the sample covariance with denominator (n - 1)
      - df = (p - 1) * (p + 2) / 2
      - rho = 1 - (2*p*p + p + 2) / (6 * (n - 1) * p)  (Bartlett factor)
      - If use_bartlett_correction is False, set rho = 1.

    Inputs
    ------
    X : array-like (n, p), optional
        Raw data. If provided, S is ignored.
    S : array-like (p, p), optional
        Sample covariance computed with denominator (n - 1). If provided, pass n.
    n : int, optional
        Sample size used to compute S when S is provided.
    center : bool, default True
        If X is provided, subtract column means before covariance.
    use_bartlett_correction : bool, default True
        Whether to apply Bartlett small-sample correction.

    Returns
    -------
    dict with keys:
        W : float
        logW : float
        T : float                # chi-square statistic
        df : int
        p_value : float
        rho : float
        n, p : ints
        sign_logdetS, logdetS, trS, t1 : diagnostics

    Notes
    -----
    The LRT requires S to be positive definite. In practice this needs p < n
    for sample covariance of full-rank data. If S is singular, the test is not
    defined (logdet is -inf).
    """
    if X is None and S is None:
        raise ValueError("Provide either X or S.")
    if S is not None and n is None:
        raise ValueError("If S is provided, also provide n (sample size).")

    if X is not None:
        X = validate_data_matrix(X)
        if X.ndim != 2:
            raise ValueError("X must be 2D.")
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

    return {
        "stat": T,
        "p_value": float(pval),
    }


def czz_sphericity_test(X, center=False):
    """
    Chen–Zhang–Zhong (2010) sphericity test for H0: Sigma = sigma^2 * I_p.

    This implementation follows the paper's location-invariant U-statistics:
      - T1 = Y1 - Y3 estimates tr(Sigma)
      - T2 = Y2 - 2*Y4 + Y5 estimates tr(Sigma^2)
      - U_n = p * (T2 / T1^2) - 1
      - Z = (n * U_n) / 2 ~ N(0, 1) under H0, so p = 1 - Phi(Z) (right-tailed)

    Inputs
    ------
    X : array-like, shape (n, p)
        Data matrix with rows as observations.
    center : bool, default False
        If True, subtract column means before computing the test.
        The original test is designed to be location-invariant without centering.

    Returns
    -------
    result : dict
        Keys:
        - U : float, the CZZ sphericity statistic U_n
        - Z : float, normal approximation statistic
        - p_normal : float, right-tailed p-value = 1 - Phi(Z)
        - n, p : ints
        - T1, T2 : floats, unbiased estimates of tr(Sigma) and tr(Sigma^2)
        - Y1, Y2, Y3, Y4, Y5 : floats, intermediate U-statistics
        - diagnostics: s_off, sum_R2, sumsq_off for verification

    Notes
    -----
    Definitions use ordered tuples of distinct indices with Pk_n = n*(n-1)*...*(n-k+1):
      Y1 = (1/n) * sum_i <X_i, X_i>
      Y2 = (1/P2_n) * sum_{i!=j} <X_i, X_j>^2
      Y3 = (1/P2_n) * sum_{i!=j} <X_i, X_j>
      Y4 = (1/P3_n) * sum_{i,j,k all distinct} <X_i, X_j> * <X_i, X_k>
      Y5 = (1/P4_n) * sum_{i,j,k,l all distinct} <X_i, X_j> * <X_k, X_l>

    Efficient computation uses the Gram matrix G = X X^T:
      - s_off = sum_{i!=j} G_ij
      - sumsq_off = sum_{i!=j} G_ij^2
      - R_i = sum_{j!=i} G_ij (off-diagonal row sums)
      - sum_R2 = sum_i R_i^2
      Then:
        sum_{i,j,k all distinct} <X_i,X_j><X_i,X_k> = sum_R2 - sumsq_off
        sum_{i,j,k,l all distinct} <X_i,X_j><X_k,X_l> = s_off^2 - 4*sum_R2 + 2*sumsq_off
    """
    X = validate_data_matrix(X)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array (n, p).")
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

    return {"stat": float(Z), "p_value": p_norm}


def hallin_rank_sphericity_test(X, method="wilcoxon"):
    """
    Van der Waerden (normal-score) rank-based test for sphericity.
    """
    X = validate_data_matrix(X)
    if method == "wilcoxon":
        n, k = X.shape
        U, d = hallin2006._center_and_scale(X)
        ranks = stats.rankdata(d, method="average")
        u = ranks / (n + 1)
        scores = u - 0.5
        score_var = 1 / 12

        Q = hallin2006._compute_statistic(U, scores, k, score_var)
        df = k * (k + 1) // 2 - 1
        pval = 1 - stats.chi2.cdf(Q, df)
    elif method == "vdw":
        n, k = X.shape
        U, d = hallin2006._center_and_scale(X)
        ranks = stats.rankdata(d, method="average")
        u = ranks / (n + 1)
        u = np.clip(u, 1e-6, 1 - 1e-6)

        raw_scores = stats.chi.ppf(u, df=k)
        scores = raw_scores - np.mean(raw_scores)

        score_var = np.var(raw_scores, ddof=0)

        Q = hallin2006._compute_statistic(U, scores, k, score_var)
        df = k * (k + 1) // 2 - 1
        pval = 1 - stats.chi2.cdf(Q, df)
    else:
        raise ValueError("Unrecognized method %s" % method)

    return {"stat": Q, "p_value": pval}
