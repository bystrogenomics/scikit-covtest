import numpy as np
from typing import Optional
from numpy.linalg import eigh
from scipy.stats import chi2


def _cov_mle(X: np.ndarray) -> np.ndarray:
    """
    MLE covariance for multivariate normal: scatter / n (ddof = 0).
    X: shape (n, p)
    """
    X = np.asarray(X, float)
    n = X.shape[0]
    Xc = X - X.mean(axis=0, keepdims=True)
    return (Xc.T @ Xc) / n


def _slogdet_safe(S: np.ndarray) -> float:
    """Return log|S| using slogdet with a positive-definiteness check."""
    sign, logdet = np.linalg.slogdet(S)
    if sign <= 0:
        raise np.linalg.LinAlgError("Covariance not positive definite.")
    return logdet


def flury_pcm(
    S_list,
    n_list,
    *,
    max_iter: int = 1000,
    tol: float = 1e-9,
    init_c: Optional[np.ndarray] = None,
    ridge: float = 0.0,
    return_history: bool = False,
):
    """
    Proportional covariance MLEs via Flury's PCM iteration.

    Model: Sigma_i = c_i * B * diag(lambda) * B^T, i = 1..k, with c_1 = 1 for identifiability.

    Parameters
    ----------
    S_list : list of (p, p) arrays
        Sample covariance MLEs, i.e., S_i = scatter_i / n_i, so that n_i S_i ~ W_p(n_i, Sigma_i).
        If you have raw data X_i, compute S_i with ddof = 0 using _cov_mle.
    n_list : list or array of ints
        Sample sizes for each group. Must be same length as S_list.
    max_iter : int
        Maximum PCM iterations.
    tol : float
        Convergence tolerance on relative changes of (c, lambda).
    init_c : array or None
        Optional initialization for c (length k). If None, initializes to ones.
        The algorithm enforces c[0] = 1 at each step by rescaling (and compensating in lambda).
    ridge : float
        Optional ridge added to each S_i as ridge * trace(S_i)/p to improve conditioning.
    return_history : bool
        If True, also return a list of dicts with iteration traces.

    Returns
    -------
    result : dict
        Keys:
            'B'        : (p, p) orthonormal eigenvectors
            'lambda'   : (p,) positive eigenvalues (lambda)
            'c'        : (k,) proportionality constants with c[0] = 1
            'iters'    : iteration count
            'converged': bool
    history : list of dicts (optional)
        Iteration diagnostics if return_history is True.
    """
    k = len(S_list)
    p = S_list[0].shape[0]
    n_list = np.asarray(n_list, dtype=float)
    n = float(np.sum(n_list))
    r = n_list / n

    # Regularize and check shapes
    S_proc = []
    for S in S_list:
        S = np.asarray(S, float)
        if S.shape != (p, p):
            raise ValueError(
                "All covariance matrices must have shape (p, p) with common p."
            )
        if ridge > 0.0:
            S = S + ridge * np.trace(S) / p * np.eye(p)
        S_proc.append(S)

    # Initialize c and lambda
    if init_c is None:
        c = np.ones(k, dtype=float)
    else:
        c = np.asarray(init_c, float).copy()
        if c.shape != (k,):
            raise ValueError("init_c must have shape (k,).")
        if c[0] <= 0:
            raise ValueError("init_c[0] must be positive.")
        # Normalize so c[0] = 1 by scaling all c and lambda accordingly later
        c = c / c[0]

    # Start with B from pooled covariance and lambda from its diagonal in that basis
    S_star = sum(r[i] * S_proc[i] / c[i] for i in range(k))
    w, B = eigh(S_star)  # ascending
    w = w[::-1]
    B = B[:, ::-1]
    # Per-axis variances a_{ij} = β_j^T S_i β_j
    A_diag = np.stack(
        [np.sum((B.T @ S_proc[i]) * B.T, axis=1) for i in range(k)], axis=0
    )  # (k, p)
    lam = np.sum((r[:, None] / c[:, None]) * A_diag, axis=0)  # lambda_j

    history = []
    converged = False

    for it in range(1, max_iter + 1):
        # PCM1: update B from weighted matrix
        S_star = sum(r[i] * S_proc[i] / c[i] for i in range(k))
        w, B = eigh(S_star)
        w = w[::-1]
        B = B[:, ::-1]

        # Compute a_{ij} on the updated B
        A_diag = np.stack(
            [np.sum((B.T @ S_proc[i]) * B.T, axis=1) for i in range(k)], axis=0
        )  # (k, p)

        # PCM2: update lambda
        lam_new = np.sum((r[:, None] / c[:, None]) * A_diag, axis=0)

        # PCM3: update c
        denom = np.maximum(lam_new, 1e-300)  # guard division
        c_new = np.mean(A_diag / denom, axis=1)  # shape (k,)

        # Renormalize to enforce c[0] = 1 while preserving Sigma_i by compensating lambda
        scale = c_new[0]
        if scale <= 0:
            raise RuntimeError("Nonpositive c[0] encountered during iteration.")
        c_new = c_new / scale
        lam_new = lam_new * scale

        # Convergence check on c and lambda
        rel_c = np.max(np.abs(c_new - c) / np.maximum(1.0, np.abs(c)))
        rel_l = np.max(np.abs(lam_new - lam) / np.maximum(1.0, np.abs(lam)))
        if return_history:
            history.append(
                {
                    "iter": it,
                    "max_rel_change_c": float(rel_c),
                    "max_rel_change_lambda": float(rel_l),
                    "min_lambda": float(np.min(lam_new)),
                    "max_lambda": float(np.max(lam_new)),
                }
            )

        c, lam = c_new, lam_new

        if max(rel_c, rel_l) < tol:
            converged = True
            break

    result = {
        "B": B,
        "lambda": lam,
        "c": c,
        "iters": it,
        "converged": converged,
    }
    if return_history:
        return result, history
    return result


def flury_proportionality_test_from_cov(
    S_list,
    n_list,
    *,
    max_iter: int = 1000,
    tol: float = 1e-9,
    ridge: float = 0.0,
    init_c: Optional[np.ndarray] = None,
):
    """
    Likelihood-ratio test for proportional covariance matrices using
    Flury's PCM MLEs.

    H0: Sigma_i = c_i Sigma_1 for i = 2..k, equivalently Sigma_i = c_i B
        diag(lambda) B^T with c_1 = 1.

    Inputs are covariance MLEs S_i with n_i S_i ~ W_p(n_i, Sigma_i).

    Returns
    -------
    out : dict
        'stat'        : LRT statistic X2
        'df'          : degrees of freedom
        'pvalue'      : chi-square upper-tail p-value
        'B','lambda','c' : MLEs under H0
        'converged'   : PCM convergence flag
        'iters'       : iterations used
    """
    k = len(S_list)
    p = S_list[0].shape[0]
    n_list = np.asarray(n_list, dtype=float)
    n = float(np.sum(n_list))

    fit = flury_pcm(
        S_list,
        n_list,
        max_iter=max_iter,
        tol=tol,
        init_c=init_c,
        ridge=ridge,
        return_history=False,
    )
    B, lam, c = fit["B"], fit["lambda"], fit["c"]

    # LRT: X^2 = n * sum_j log lambda_j + sum_i n_i (p log c_i - log |S_i|)
    logdet_S = np.array([_slogdet_safe(S) for S in S_list])
    X2 = n * float(np.sum(np.log(np.maximum(lam, 1e-300)))) + float(
        np.sum(n_list * (p * np.log(np.maximum(c, 1e-300)) - logdet_S))
    )

    df = (k - 1) * (p * (p + 1) - 2) // 2
    pval = chi2.sf(X2, df)

    return {
        "stat": X2,
        "df": df,
        "p_value": pval,
        "B": B,
        "lambda": lam,
        "c": c,
        "converged": fit["converged"],
        "iters": fit["iters"],
    }


def flury_proportionality_test(
    X_list,
    *,
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
        Each X_i has shape (n_i, p). This function computes
        S_i = scatter / n_i.

    Returns
    -------
    out : dict
        See flury_proportionality_test_from_cov.
    """
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
        S_list.append(_cov_mle(X))
        n_list.append(n_i)
    return flury_proportionality_test_from_cov(
        S_list, n_list, max_iter=max_iter, tol=tol, ridge=ridge, init_c=init_c
    )
