import numpy as np


def _validate_matrix(X, name):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"{name} must be 2D with shape (n_samples, n_features).")
    if not np.isfinite(X).all():
        raise ValueError(f"{name} contains NaN/Inf.")
    return X


def _center(X):
    return X - X.mean(axis=0, keepdims=True)


def _cov_and_theta_hat(Xc):
    """
    Chang et al. use:
      - sample covariance:  S = (1/n) sum_i x_i x_i^T  (with x_i centered)
      - variance estimator: theta_{jk} = (1/n) sum_i (x_{ij}x_{ik} - S_{jk})^2

    We compute theta efficiently via:
      E[(x_j x_k)^2] - (E[x_j x_k])^2
    where E[(x_j x_k)^2] = (1/n) sum_i x_{ij}^2 x_{ik}^2.

    Returns
    -------
    S : (p,p)
    theta : (p,p)
    """
    n, p = Xc.shape

    # S_{jk} = (1/n) sum_i x_{ij} x_{ik}
    S = (Xc.T @ Xc) / n

    # E[(x_j x_k)^2] = (1/n) sum_i x_{ij}^2 x_{ik}^2
    Xsq = Xc * Xc
    E_sq = (Xsq.T @ Xsq) / n

    theta = E_sq - (S * S)
    # Numerical guard (finite-sample can yield tiny negative due to roundoff)
    theta = np.maximum(theta, 0.0)
    return S, theta


def chang_2017_perturbation_max_test(
    X,
    Y,
    alpha=0.05,
    B=1000,
    use_upper_triangle=True,
    eps=1e-12,
    random_state=None,
):
    """
    Chang et al. (Biometrics 2017) perturbation (multiplier bootstrap) max test
    for equality of two covariance matrices in high dimension.

    Inputs
    ------
    X : array (n, p)  First group, rows = samples
    Y : array (m, p)  Second group, rows = samples
    alpha : significance level for critical value
    B : number of multiplier bootstrap replicates
    use_upper_triangle : if True, max over i<=j; else over all i,j
    eps : jitter for denominator stability
    random_state : None | int | np.random.Generator

    Returns
    -------
    dict with:
      - Tmax : observed max statistic
      - critical_value : bootstrap (1-alpha) quantile of Tmax^*
      - p_value : bootstrap p-value (upper tail)
      - reject : Tmax > critical_value
      - n, m, p
    """
    X = _validate_matrix(X, "X")
    Y = _validate_matrix(Y, "Y")
    n, p = X.shape
    m, p2 = Y.shape
    if p2 != p:
        raise ValueError("X and Y must have the same number of features (columns).")
    if n < 2 or m < 2:
        raise ValueError("Need at least 2 samples per group.")

    rng = random_state
    if isinstance(rng, (int, np.integer)) or rng is None:
        rng = np.random.default_rng(rng)
    elif not isinstance(rng, np.random.Generator):
        raise ValueError("random_state must be None, an int seed, or a numpy.random.Generator.")

    # Center data
    Xc = _center(X)
    Yc = _center(Y)

    # Sample covariances and theta-hats (variance of centered cross-products)
    Sx, thetax = _cov_and_theta_hat(Xc)
    Sy, thetay = _cov_and_theta_hat(Yc)

    # Denominator matrix for t_{jk}
    denom = thetax / n + thetay / m
    denom = np.maximum(denom, eps)

    # Observed entrywise t-stats and Tmax
    T = (Sx - Sy) / np.sqrt(denom)
    if use_upper_triangle:
        iu = np.triu_indices(p)
        Tmax = float(np.max(np.abs(T[iu])))
    else:
        Tmax = float(np.max(np.abs(T)))

    # Multiplier bootstrap for critical value / p-value
    Tmax_star = np.empty(B, dtype=float)

    # Precompute for speed
    # (We use the paper's perturbation: (1/n) sum g_i (x_i x_i^T - S))
    for b in range(B):
        g1 = rng.standard_normal(n)
        g2 = rng.standard_normal(m)

        # Efficient weighted outer sums:
        # Sum_i g_i x_i x_i^T  = X^T (diag(g) X) = X^T (g[:,None]*X)
        Wx = Xc * g1[:, None]
        Wy = Yc * g2[:, None]

        sum_g1 = float(g1.sum())
        sum_g2 = float(g2.sum())

        Sx_star = (Xc.T @ Wx) / n - Sx * (sum_g1 / n)
        Sy_star = (Yc.T @ Wy) / m - Sy * (sum_g2 / m)

        T_star = (Sx_star - Sy_star) / np.sqrt(denom)

        if use_upper_triangle:
            Tmax_star[b] = np.max(np.abs(T_star[iu]))
        else:
            Tmax_star[b] = np.max(np.abs(T_star))

    critical_value = float(np.quantile(Tmax_star, 1.0 - alpha))

    # Upper-tail bootstrap p-value (add-one smoothing)
    p_value = float((1.0 + np.sum(Tmax_star >= Tmax)) / (B + 1.0))
    reject = bool(Tmax > critical_value)

    return {
        "Tmax": Tmax,
        "critical_value": critical_value,
        "p_value": p_value,
        "reject": reject,
        "n": int(n),
        "m": int(m),
        "p": int(p),
        "B": int(B),
        "alpha": float(alpha),
    }


# ---- quick smoke test ----
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n, m, p = 40, 45, 80
    X = rng.normal(size=(n, p))
    Y = rng.normal(size=(m, p))
    out = chang_2017_perturbation_max_test(X, Y, alpha=0.05, B=300, random_state=1)
    print(out)

