import numpy as np
from numpy.linalg import inv, norm
from scipy.stats import chi2

# ---------- Core helpers from the unadjusted LRT ----------


def _sample_cov(X):
    """Unbiased sample covariance with ddof=1; rows are observations."""
    return np.cov(X, rowvar=False, ddof=1)


def _ensure_pd(A, ridge):
    """Add a small ridge if needed to ensure positive-definiteness."""
    if ridge <= 0:
        return A
    p = A.shape[0]
    return A + ridge * np.eye(p)


def fit_proportional_covariances(
    S_list, n_list, *, tol=1e-10, max_iter=10_000, ridge=0.0
):
    """
    MLE under proportionality: Sigma_k = c_k * Sigma  (k = 1..K).
    """
    K = len(S_list)
    p = S_list[0].shape[0]
    n_list = np.asarray(n_list, dtype=float)
    assert all(S.shape == (p, p) for S in S_list)
    assert len(n_list) == K

    S_list = [_ensure_pd(S.astype(float), ridge) for S in S_list]
    n_plus = float(np.sum(n_list))

    # Start at pooled covariance
    Sigma = sum(n_list[k] * S_list[k] for k in range(K)) / n_plus
    Sigma = _ensure_pd(Sigma, ridge)

    for it in range(1, max_iter + 1):
        Sigma_inv = inv(Sigma)
        c = np.array([np.trace(Sigma_inv @ S_list[k]) / p for k in range(K)])
        Sigma_new = (
            sum(n_list[k] * (S_list[k] / c[k]) for k in range(K)) / n_plus
        )
        Sigma_new = _ensure_pd(Sigma_new, ridge)

        rel = norm(Sigma_new - Sigma, ord="fro") / max(
            1e-16, norm(Sigma, ord="fro")
        )
        Sigma = Sigma_new
        if rel < tol:
            return Sigma, c, True, it

    return Sigma, c, False, max_iter


def _wilks_stat_from_S(S_list, n_list, Sigma_hat, c_hat, ridge=0.0):
    """
    Compute -2 log Lambda given S_k, n_k, and the MLEs under H0 (Sigma_hat, c_hat).
    """
    n_arr = np.asarray(n_list, dtype=float)
    p = S_list[0].shape[0]
    n_plus = float(np.sum(n_arr))

    sign_Sigma, logdet_Sigma = np.linalg.slogdet(_ensure_pd(Sigma_hat, ridge))
    if sign_Sigma <= 0:
        raise np.linalg.LinAlgError("hat(Sigma) not PD.")

    term1 = n_plus * logdet_Sigma
    term2 = 0.0
    for k, S in enumerate(S_list):
        sign_Sk, logdet_Sk = np.linalg.slogdet(_ensure_pd(S, ridge))
        if sign_Sk <= 0:
            raise np.linalg.LinAlgError("S_k not PD.")
        term2 += n_arr[k] * logdet_Sk

    if np.any(c_hat <= 0):
        raise ValueError("Nonpositive c_k encountered.")
    term3 = p * np.sum(n_arr * np.log(c_hat))
    return term1 - term2 + term3  # -2 log Λ


def test_cov_proportionality(
    X_groups=None,
    *,
    S_list=None,
    n_list=None,
    ridge=0.0,
    tol=1e-10,
    max_iter=10_000,
):
    """
    Unadjusted Wilks LRT for H0: Sigma_k = c_k Sigma.
    """
    if X_groups is None and (S_list is None or n_list is None):
        raise ValueError(
            "Provide either X_groups, or S_list together with n_list."
        )

    if X_groups is not None:
        S_list = [_sample_cov(X) for X in X_groups]
        n_list = [X.shape[0] - 1 for X in X_groups]

    K = len(S_list)
    p = S_list[0].shape[0]

    Sigma_hat, c_hat, converged, iters = fit_proportional_covariances(
        S_list, n_list, tol=tol, max_iter=max_iter, ridge=ridge
    )
    stat = _wilks_stat_from_S(S_list, n_list, Sigma_hat, c_hat, ridge=ridge)

    # df = (K−1)[ p(p+1)/2 − 1 ]
    df = int((K - 1) * (p * (p + 1) // 2 - 1))
    pval = 1.0 - chi2.cdf(stat, df)

    return dict(
        stat=float(stat),
        df=df,
        pvalue=float(pval),
        Sigma_hat=Sigma_hat,
        c_hat=c_hat,
        converged=converged,
        iterations=iters,
        n_list=np.asarray(n_list, dtype=float),
        S_list=[np.array(S, dtype=float) for S in S_list],
    )


# ---------- Bartlett-adjusted LRT (parametric factor) ----------


def bartlett_adjusted_proportionality_test(
    X_groups=None,
    *,
    S_list=None,
    n_list=None,
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
    1) Fit H0 by MLE (hat(Sigma), hat{c}) using the proportional model
       Sigma_k = c_k Sigma.
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
    S_list, n_list : provide these if X_groups is not given.
        S_list: unbiased sample covariances (ddof=1), n_list: n_k = N_k - 1.
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
        - 'phi'         : multiplicative Bartlett factor
                          (approx df / E[T] under H0)
        - 'stat_adj'    : Bartlett-adjusted statistic  (phi * stat)
        - 'pvalue_adj'  : Bartlett-adjusted p-value    (χ²_df right-tail)
        - 'Sigma_hat', 'c_hat', 'converged', 'iterations', 'n_list','S_list'
    """
    rng = np.random.default_rng(random_state)

    # 1) Unadjusted fit and statistic
    base = test_cov_proportionality(
        X_groups=X_groups,
        S_list=S_list,
        n_list=n_list,
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
        Sk = _ensure_pd(Sk, ridge if ridge > 0 else 1e-12)
        chol_list.append(np.linalg.cholesky(Sk))

    # 3) Parametric bootstrap to estimate E[T] under H0 and phi
    T_boot = np.empty(B, dtype=float)
    for b in range(B):
        S_b = []
        for k in range(K):
            n_k = int(n_list[k])
            # Simulate n_k + 1 observations so that ddof=1 yields n_k in Wishart
            Xk = rng.standard_normal((n_k + 1, p)) @ chol_list[k].T
            S_b.append(_sample_cov(Xk))
        if refit_mle_each_boot:
            Sigma_b, c_b, _, _ = fit_proportional_covariances(
                S_b, n_list, tol=tol, max_iter=max_iter, ridge=ridge
            )
            T_b = _wilks_stat_from_S(S_b, n_list, Sigma_b, c_b, ridge=ridge)
        else:
            T_b = _wilks_stat_from_S(S_b, n_list, Sigma_hat, c_hat, ridge=ridge)
        T_boot[b] = T_b

    mean_T = float(np.mean(T_boot))
    # Multiplicative Bartlett factor so that E[phi*T] approx df
    phi = df / mean_T if mean_T > 0 else 1.0

    T_adj = phi * T_obs
    p_adj = 1.0 - chi2.cdf(T_adj, df)

    return {
        **base,
        "phi": float(phi),
        "stat_adj": float(T_adj),
        "pvalue_adj": float(p_adj),
    }
