import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import LinAlgError
from scipy.linalg import eigh
from scipy.stats import chi2, lognorm, norm


def _center(X):
    return X - np.mean(X, axis=0, keepdims=True)


def _cov(X, ddof=1):
    n = X.shape[0]
    return (X.T @ X) / (n - ddof)


def _mahalanobis2(X, mu, Sigma_inv):
    D = X - mu
    return np.sum(D @ Sigma_inv * D, axis=1)


def _remove_top_right_spines(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)


def eigen_spectrum(
    X,
    center=True,
    ddof=1,
    sort_desc=True,
    plot=True,
    show=True,
    title="Eigenvalue spectrum",
    overlay_mp=True,
):
    """
    Compute and optionally plot eigenvalue spectrum of the sample covariance.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Input data matrix. Rows are samples, columns are features.

    center : bool, default=True
        If True, subtract the mean of each column before computing
        the covariance.

    ddof : int, default=1
        Delta degrees of freedom used in covariance estimation,
        passed to denominator ``n - ddof``.

    sort_desc : bool, default=True
        If True, return eigenvalues sorted in descending order.

    plot : bool, default=True
        If True, plot the eigenvalue spectrum.

    show : bool, default=True
        If True and ``plot=True``, display the plot immediately.

    title : str, default="Eigenvalue spectrum"
        Title for the eigenvalue spectrum plot.

    overlay_mp : bool, default=True
        If True, overlay Marchenko–Pastur bulk edges as horizontal
        dashed lines.

    Returns
    -------
    results : dict
        Dictionary with keys:
        - 'eigenvalues' : ndarray
          Array of eigenvalues.
        - 'mp_min', 'mp_max', 'mp_q' : float, optional
          Marchenko–Pastur bounds and aspect ratio if overlay requested.

    See Also
    --------
    condition_and_rank : Compute condition number and effective rank.
    """
    Xc = _center(X) if center else X
    S = _cov(Xc, ddof=ddof)
    # symmetric eigen-decomposition
    w = eigh(S, eigvals_only=True)
    if sort_desc:
        w = np.sort(w)[::-1]

    out = {"eigenvalues": w}

    if plot:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(np.arange(1, len(w) + 1), w, marker="o", linewidth=2)
        ax.set_xlabel("Index", fontsize=12)
        ax.set_ylabel("Eigenvalue", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        _remove_top_right_spines(ax)

        if overlay_mp:
            # Marchenko–Pastur bulk edges assuming white noise with variance = mean eigenvalue
            n, p = X.shape
            q = p / max(n - ddof, 1)
            mean_eig = np.mean(w)
            if q > 0:
                mp_min = mean_eig * (1 - np.sqrt(q)) ** 2
                mp_max = mean_eig * (1 + np.sqrt(q)) ** 2
                out["mp_min"], out["mp_max"], out["mp_q"] = mp_min, mp_max, q
                ax.axhline(mp_min, linestyle="--", linewidth=2)
                ax.axhline(mp_max, linestyle="--", linewidth=2)
        plt.tight_layout()
        if show:
            plt.show()

    return out


def mardia_tests(data, use_population=True, tol=1e-25, bootstrap=False, B=1000):
    """
    Compute Mardia's multivariate skewness and kurtosis tests.

    Parameters
    ----------
    data : ndarray (n × p)
        Observations (rows = samples, columns = variables).
    use_population : bool, default=True
        Whether to use the population covariance (dividing by n) instead of the sample covariance (dividing by n-1).
    tol : float, default=1e-25
        Tolerance for covariance inversion.
    bootstrap : bool, default=False
        If True, perform parametric bootstrap for p-values.
    B : int, default=1000
        Number of bootstrap replicates.

    Returns
    -------
    result : dict
        Contains test names, statistics, p-values, and method info.
    """

    # --- Basic validation ---
    x = np.asarray(data, dtype=float)
    n, p = x.shape
    if p < 2:
        raise ValueError(
            "Need at least two numeric variables for Mardia's test."
        )
    if np.isnan(x).any():
        mask = ~np.isnan(x).any(axis=1)
        x = x[mask]
        n = x.shape[0]
        print(
            f"Warning: {np.sum(~mask)} rows with missing values were removed."
        )

    # --- Center and covariance ---
    x_centered = x - np.mean(x, axis=0)
    S = np.cov(x_centered, rowvar=False, bias=use_population)

    # --- Invert covariance ---
    try:
        invS = np.linalg.inv(S)
    except LinAlgError as e:
        raise ValueError(f"Covariance matrix is singular or near-singular: {e}")

    # --- Mahalanobis distance matrix ---
    D = x_centered @ invS @ x_centered.T

    # --- Mardia’s measures (observed) ---
    g1p_obs = np.sum(D**3) / n**2
    g2p_obs = np.sum(np.diag(D) ** 2) / n

    # --- Skewness test ---
    df_skew = p * (p + 1) * (p + 2) / 6
    k_const = ((p + 1) * (n + 1) * (n + 3)) / (n * ((n + 1) * (p + 1) - 6))
    skew_stat = n * k_const * g1p_obs / 6 if n < 20 else n * g1p_obs / 6
    p_skew = 1 - chi2.cdf(skew_stat, df_skew)

    # --- Kurtosis test ---
    kurt_stat = (g2p_obs - p * (p + 2)) * np.sqrt(n / (8 * p * (p + 2)))
    p_kurt = 2 * (1 - norm.cdf(abs(kurt_stat)))

    result = {
        "Test": ["Mardia Skewness", "Mardia Kurtosis"],
        "stat": [skew_stat, kurt_stat],
        "p_value": [p_skew, p_kurt],
        "Method": ["asymptotic", "asymptotic"],
    }

    # --- Optional bootstrap ---
    if bootstrap and B > 0:
        mu_hat = np.mean(x, axis=0)
        Sigma_hat = S
        sq_factor = np.sqrt(n / (8 * p * (p + 2)))

        skew_boot, kurt_boot = [], []
        for _ in range(B):
            try:
                xb = np.random.multivariate_normal(mu_hat, Sigma_hat, size=n)
                xb_c = xb - np.mean(xb, axis=0)
                Sb = np.cov(xb_c, rowvar=False, bias=use_population)
                invSb = np.linalg.inv(Sb)
                Db = xb_c @ invSb @ xb_c.T
                Djb = np.diag(Db)

                g1p_b = np.sum(Db**3) / n**2
                skew_b = n * k_const * g1p_b / 6 if n < 20 else n * g1p_b / 6

                g2p_b = np.sum(Djb**2) / n
                kurt_b = (g2p_b - p * (p + 2)) * sq_factor

                skew_boot.append(skew_b)
                kurt_boot.append(kurt_b)
            except LinAlgError:
                continue

        if len(skew_boot) > 0:
            skew_boot = np.array(skew_boot)
            kurt_boot = np.array(kurt_boot)
            result["p.value"] = [
                np.mean(skew_boot >= skew_stat),
                np.mean(np.abs(kurt_boot) >= abs(kurt_stat)),
            ]
            result["Method"] = ["bootstrap", "bootstrap"]

    return result


def shapiro_francia_w(x):
    """
    Compute the Shapiro–Francia W statistic for a 1D array x.
    Based on Royston (1992) and original formula.
    """
    n = len(x)
    if n < 3:
        raise ValueError(
            "Sample size must be at least 3 for Shapiro–Francia W."
        )

    x = np.asarray(x)
    x_sorted = np.sort(x)
    m = norm.ppf((np.arange(1, n + 1) - 3 / 8) / (n + 1 / 4))  # Blom's formula
    m /= np.linalg.norm(m)

    w = (np.sum(m * x_sorted)) ** 2 / np.sum((x - np.mean(x)) ** 2)
    return w


def royston_test(data):
    """
    Royston's multivariate normality test (Python version matching MVN::roystonTest).

    Parameters
    ----------
    data : ndarray (n × p)
        Input data array.

    Returns
    -------
    dict
        {'Test', 'Statistic', 'p.value', 'Method'}
    """
    x = np.asarray(data, dtype=float)
    n, p = x.shape
    if p < 2:
        raise ValueError(
            "Need at least two numeric variables for Royston's test."
        )
    if n <= 3 or n > 2000:
        raise ValueError("Sample size must be >3 and <=2000.")

    # Compute univariate W-statistics and z-transforms
    z_vals = np.zeros(p)
    if 4 <= n <= 11:
        # Small sample constants
        g = -2.273 + 0.459 * n
        m = 0.544 - 0.39978 * n + 0.025054 * n**2 - 0.0006714 * n**3
        s = np.exp(
            1.3822 - 0.77857 * n + 0.062767 * n**2 - 0.0020322 * n**3
        )
        small_sample = True
    else:
        # Large sample constants
        lx = np.log(n)
        m = -1.5861 - 0.31082 * lx - 0.083751 * lx**2 + 0.0038915 * lx**3
        s = np.exp(-0.4803 - 0.082676 * lx + 0.0030302 * lx**2)
        small_sample = False

    # Compute z_i for each variable
    for i in range(p):
        vec = x[:, i]
        w = shapiro_francia_w(vec)
        if small_sample:
            z_vals[i] = (-np.log(g - np.log(1 - w)) - m) / s
        else:
            z_vals[i] = (np.log(1 - w) - m) / s

    # Correlation adjustment
    u = 0.715
    v = 0.21364 + 0.015124 * (np.log(n) ** 2) - 0.0018034 * (np.log(n) ** 3)
    ll = 5
    C = np.corrcoef(x, rowvar=False)
    NC = (C**ll) * (1 - (u * (1 - C) ** u) / v)
    T = np.sum(NC) - p
    mC = T / (p**2 - p)
    edf = p / (1 + (p - 1) * mC)

    # Observed H statistic
    H_stat = (edf * np.sum((norm.ppf(norm.cdf(-z_vals) / 2)) ** 2)) / p

    # Asymptotic p-value
    p_value = 1 - chi2.cdf(H_stat, df=edf)

    return {
        "Test": "Royston",
        "stat": H_stat,
        "p_value": p_value,
        "Method": "asymptotic",
    }


def hz_test(data, use_population=True, tol=1e-25, bootstrap=False, B=1000):
    """
    Henze–Zirkler test for multivariate normality.

    Parameters
    ----------
    data : ndarray (n × p)
        Input data matrix.
    use_population : bool, default=True
        Whether to use the population covariance (divide by n) or sample covariance (divide by n-1).
    tol : float, default=1e-25
        Tolerance for covariance inversion.
    bootstrap : bool, default=False
        Whether to use a parametric bootstrap for the p-value.
    B : int, default=1000
        Number of bootstrap replicates (only used if bootstrap=True).

    Returns
    -------
    dict
        With test name, statistic, p-value, and method info.
    """

    # --- Validate and clean data ---
    x = np.asarray(data, dtype=float)
    n, p = x.shape
    if p < 2:
        raise ValueError(
            "Need at least two numeric variables for Henze–Zirkler test."
        )
    if np.isnan(x).any():
        mask = ~np.isnan(x).any(axis=1)
        print(
            f"Warning: {np.sum(~mask)} rows with missing values were removed."
        )
        x = x[mask]
        n = x.shape[0]

    # --- Center data and compute covariance ---
    x_centered = x - np.mean(x, axis=0)
    S = np.cov(x_centered, rowvar=False, bias=use_population)

    # --- Invert covariance ---
    try:
        invS = np.linalg.inv(S)
    except np.linalg.LinAlgError as e:
        raise ValueError(f"Covariance matrix is singular or near-singular: {e}")

    # --- Mahalanobis distances ---
    D = x_centered @ invS @ x_centered.T
    Dj = np.diag(D)
    Djk = np.add.outer(Dj, Dj) - 2 * D  # pairwise squared distances

    # --- Smoothing parameter b ---
    b = (n ** (1 / (p + 4))) * (((2 * p + 1) / 4) ** (1 / (p + 4))) / np.sqrt(2)

    # --- HZ statistic ---
    part1 = np.sum(np.exp(-(b**2) / 2 * Djk)) / (n**2)
    part2 = (
        2
        * (1 + b**2) ** (-p / 2)
        * np.sum(np.exp(-(b**2) / (2 * (1 + b**2)) * Dj))
        / n
    )
    hz_stat = n * (part1 - part2 + (1 + 2 * b**2) ** (-p / 2))

    # --- Log-normal approximation parameters ---
    a = 1 + 2 * b**2
    wb = (1 + b**2) * (1 + 3 * b**2)
    mu = 1 - a ** (-p / 2) * (
        1 + (p * b**2) / a + (p * (p + 2) * b**4) / (2 * a**2)
    )
    si2 = (
        2 * (1 + 4 * b**2) ** (-p / 2)
        + 2
        * a ** (-p)
        * (
            1
            + (2 * p * b**4) / a**2
            + (3 * p * (p + 2) * b**8) / (4 * a**4)
        )
        - 4
        * wb ** (-p / 2)
        * (
            1
            + (3 * p * b**4) / (2 * wb)
            + (p * (p + 2) * b**8) / (2 * wb**2)
        )
    )
    pmu = np.log(np.sqrt(mu**4 / (si2 + mu**2)))
    psi = np.sqrt(np.log((si2 + mu**2) / mu**2))

    # --- Asymptotic p-value ---
    p_value = lognorm.sf(hz_stat, s=psi, scale=np.exp(pmu))

    method = "asymptotic"

    # --- Bootstrap option ---
    if bootstrap and B > 0:
        mu_hat = np.mean(x, axis=0)
        Sigma_hat = S
        boot_stats = []

        for _ in range(B):
            try:
                xb = np.random.multivariate_normal(mu_hat, Sigma_hat, size=n)
                xb_c = xb - np.mean(xb, axis=0)
                Sb = np.cov(xb_c, rowvar=False, bias=use_population)
                invSb = np.linalg.inv(Sb)
                Db = xb_c @ invSb @ xb_c.T
                Djb = np.diag(Db)
                Djkb = np.add.outer(Djb, Djb) - 2 * Db

                bb = b  # same smoothing param
                part1b = np.sum(np.exp(-(bb**2) / 2 * Djkb)) / (n**2)
                part2b = (
                    2
                    * (1 + bb**2) ** (-p / 2)
                    * np.sum(np.exp(-(bb**2) / (2 * (1 + bb**2)) * Djb))
                    / n
                )
                hz_b = n * (part1b - part2b + (1 + 2 * bb**2) ** (-p / 2))
                boot_stats.append(hz_b)
            except np.linalg.LinAlgError:
                continue

        if len(boot_stats) > 0:
            boot_stats = np.array(boot_stats)
            p_value = np.mean(boot_stats >= hz_stat)
            method = "bootstrap"
        else:
            p_value = np.nan

    return {
        "Test": "Henze–Zirkler",
        "stat": hz_stat,
        "p_value": p_value,
        "Method": method,
    }


def condition_and_rank(X, center=True, ddof=1, eps=1e-12):
    """
    Compute spectral condition number, numerical rank, and effective rank.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Input data matrix.

    center : bool, default=True
        If True, subtract the mean of each column.

    ddof : int, default=1
        Delta degrees of freedom for covariance estimation.

    eps : float, default=1e-12
        Tolerance for identifying small eigenvalues in rank calculations.

    Returns
    -------
    results : dict
        - 'condition_number' : float
          Ratio of largest to smallest eigenvalue.
        - 'numerical_rank' : int
          Count of eigenvalues above tolerance threshold.
        - 'effective_rank' : float
          Entropy-based effective rank.
        - 'eigenvalues' : ndarray
          Sorted eigenvalues of the covariance matrix.
        - 'warnings' : list of str
          Human-readable warnings for unstable conditions.

    Notes
    -----
    Large condition numbers or low effective rank indicate instability
    in high-dimensional settings.
    """
    Xc = _center(X) if center else X
    n, p = Xc.shape
    S = _cov(Xc, ddof=ddof)
    # eigenvalues (nonnegative; use symmetric solver)
    w = np.sort(eigh(S, eigvals_only=True))[::-1]
    w_clipped = np.clip(w, 0.0, None)

    # condition number
    lam_max = w_clipped[0]
    lam_min = w_clipped[-1]
    cond = np.inf if lam_min <= eps else lam_max / lam_min

    # numerical rank ( relative to largest eigenvalue )
    tol = eps * max(p, n)
    num_rank = int(np.sum(w_clipped > max(lam_max * 1e-8, tol)))

    # effective rank (entropy-based)
    if w_clipped.sum() > 0:
        p_i = w_clipped / w_clipped.sum()
        entropy = -np.sum(p_i * np.log(np.clip(p_i, 1e-300, 1.0)))
        e_rank = float(np.exp(entropy))
    else:
        e_rank = 0.0

    warnings = []
    if n < p:
        warnings.append(
            "n < p: high-dimensional regime; some tests may be unstable."
        )
    if np.isinf(cond) or cond > 1e8:
        warnings.append("Ill-conditioned covariance (condition number > 1e8).")
    if e_rank < 0.5 * p:
        warnings.append(
            "Low effective rank (< 50% of p): strong collinearity or near-degeneracy."
        )
    if num_rank < p:
        warnings.append("Numerical rank deficiency detected.")

    return {
        "condition_number": cond,
        "numerical_rank": num_rank,
        "effective_rank": e_rank,
        "eigenvalues": w,
        "warnings": warnings,
    }
