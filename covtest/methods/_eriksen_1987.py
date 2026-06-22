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


def _c_from_sigma(Sigma, S_list, p, ridge=0.0):
    Sigma_inv = inv(_ensure_pd(Sigma, ridge))
    return np.array([np.trace(Sigma_inv @ _ensure_pd(S, ridge)) / p for S in S_list])


def _validate_inputs(X_groups, S_list, n_list, ridge):
    if X_groups is None and (S_list is None or n_list is None):
        raise ValueError(
            "Provide either X_groups, or S_list together with n_list."
        )

    if X_groups is not None:
        p = X_groups[0].shape[1]
        for k, X in enumerate(X_groups):
            if X.ndim != 2:
                raise ValueError(f"Group {k} must be a 2D array.")
            if X.shape[1] != p:
                raise ValueError("All groups must have the same number of features p.")
            N_k = X.shape[0]
            if N_k < p + 1:
                raise ValueError(
                    f"Group {k} must have at least p+1 observations for stability (got N={N_k}, p={p})."
                )
        S_list = [_sample_cov(X) for X in X_groups]
        n_list = [X.shape[0] - 1 for X in X_groups]
    else:
        if len(S_list) == 0:
            raise ValueError("S_list cannot be empty.")
        first_S = S_list[0]
        if first_S.ndim == 2:
            p = first_S.shape[0]
        elif first_S.ndim == 1:
            p = first_S.shape[0]
        else:
            p = 1

    K = len(S_list)

    if len(S_list) != len(n_list):
        raise ValueError("S_list and n_list must have the same length.")

    if p < 2:
        raise ValueError("p must be at least 2 (proportionality is vacuous for p=1).")

    for k, n_val in enumerate(n_list):
        if n_val <= 0:
            raise ValueError(f"n_{k} must be positive.")

    for k, S in enumerate(S_list):
        if S.ndim != 2 or S.shape[0] != S.shape[1]:
            raise ValueError(f"S_{k} must be a square matrix.")
        if S.shape[0] != p:
            raise ValueError(f"S_{k} must have shape ({p}, {p}).")
        
        # Positive definiteness check: fail early if logdet is not positive
        sign, logdet = np.linalg.slogdet(_ensure_pd(S, ridge))
        if sign <= 0:
            raise ValueError(f"Sample covariance matrix S_{k} is not positive definite.")

    return S_list, n_list, K, p


def fit_proportional_covariances(
    S_list, n_list, *, tol=1e-10, max_iter=10_000, ridge=0.0
):
    """
    MLE under proportionality: Sigma_k = c_k * Sigma  (k = 1..K).
    """
    K = len(S_list)
    if K == 0:
        raise ValueError("S_list cannot be empty.")
    first_S = S_list[0]
    if first_S.ndim == 2:
        p = first_S.shape[0]
    elif first_S.ndim == 1:
        p = first_S.shape[0]
    else:
        p = 1
    n_list = np.asarray(n_list, dtype=float)
    
    if len(S_list) != len(n_list):
        raise ValueError("S_list and n_list must have the same length.")
    if p < 2:
        raise ValueError("p must be at least 2.")
    for k, S in enumerate(S_list):
        if S.shape != (p, p):
            raise ValueError(f"S_{k} must have shape ({p}, {p}).")

    S_list = [_ensure_pd(S.astype(float), ridge) for S in S_list]
    n_plus = float(np.sum(n_list))

    # Start at pooled covariance
    Sigma = sum(n_list[k] * S_list[k] for k in range(K)) / n_plus
    Sigma = _ensure_pd(Sigma, ridge)

    converged = False
    iters = max_iter
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
            converged = True
            iters = it
            break

    c_hat = _c_from_sigma(Sigma, S_list, p, ridge)
    return Sigma, c_hat, converged, iters


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
    S_list, n_list, K, p = _validate_inputs(X_groups, S_list, n_list, ridge)

    Sigma_hat, c_hat, converged, iters = fit_proportional_covariances(
        S_list, n_list, tol=tol, max_iter=max_iter, ridge=ridge
    )
    stat = _wilks_stat_from_S(S_list, n_list, Sigma_hat, c_hat, ridge=ridge)

    # df = (K−1)[ p(p+1)/2 − 1 ]
    df = int((K - 1) * (p * (p + 1) // 2 - 1))
    pval = chi2.sf(stat, df)

    return dict(
        stat=float(stat),
        df=df,
        p_value=float(pval),
        pvalue=float(pval),  # for backward compatibility
        Sigma_hat=Sigma_hat,
        c_hat=c_hat,
        converged=converged,
        iterations=iters,
        n_list=np.asarray(n_list, dtype=float),
        S_list=[np.array(S, dtype=float) for S in S_list],
    )


# ---------- Bartlett-adjusted LRT (parametric factor) ----------


def bootstrap_bartlett_adjusted_proportionality_test(
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
    Bootstrap Bartlett-adjusted Wilks LRT for proportional covariance matrices.

    This function performs a parametric-bootstrap Bartlett-style calibration of the 
    likelihood ratio test statistic. Note that this is NOT Eriksen's closed-form analytic 
    Bartlett adjustment (Theorem 6.1 of Eriksen 1987).

    Parameters
    ----------
    X_groups : list of arrays, optional
        Observations per group (N_k x p).
    S_list, n_list : provide these if X_groups is not given.
        S_list: unbiased sample covariances (ddof=1), n_list: n_k = N_k - 1.
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
    import warnings
    if not refit_mle_each_boot:
        warnings.warn(
            "refit_mle_each_boot=False is not a true LRT bootstrap calibration "
            "because the null model is not re-maximized for each bootstrap replicate.",
            UserWarning,
            stacklevel=2,
        )

    S_list, n_list, K, p = _validate_inputs(X_groups, S_list, n_list, ridge)
    rng = np.random.default_rng(random_state)

    # 1) Unadjusted fit and statistic
    base = test_cov_proportionality(
        S_list=S_list,
        n_list=n_list,
        ridge=ridge,
        tol=tol,
        max_iter=max_iter,
    )
    Sigma_hat = base["Sigma_hat"]
    c_hat = base["c_hat"]
    df = base["df"]
    T_obs = base["stat"]
    p_unadj = base["p_value"]

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
    phi = df / mean_T if mean_T > 0 else 1.0

    T_boot_adj = phi * T_obs
    p_boot_adj = chi2.sf(T_boot_adj, df)

    return {
        "stat": float(T_boot_adj),
        "p_value": float(p_boot_adj),
        "pvalue": float(p_boot_adj),  # for backward compatibility
        "stat_unadjusted": float(T_obs),
        "p_value_unadjusted": float(p_unadj),
        "pvalue_unadjusted": float(p_unadj),  # for backward compatibility
        "df": df,
        "phi_bootstrap": float(phi),
        "stat_bootstrap_adj": float(T_boot_adj),
        "p_value_bootstrap_adj": float(p_boot_adj),
        "pvalue_bootstrap_adj": float(p_boot_adj),  # for backward compatibility
        "mean_T_bootstrap": float(mean_T),
        "B": B,
        "Sigma_hat": Sigma_hat,
        "c_hat": c_hat,
        "cov_hat_groups": [c_hat[k] * Sigma_hat for k in range(K)],
        "converged": base["converged"],
        "iterations": base["iterations"],
        "n_list": np.asarray(n_list, dtype=float),
        "S_list": S_list,
    }


def bartlett_adjusted_proportionality_test(*args, **kwargs):
    import warnings
    warnings.warn(
        "bartlett_adjusted_proportionality_test is deprecated; "
        "use bootstrap_bartlett_adjusted_proportionality_test instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return bootstrap_bartlett_adjusted_proportionality_test(*args, **kwargs)
