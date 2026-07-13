import numpy as np
import numpy.linalg as la


def robust_location(X, tol=1e-6, max_iter=200):
    """Robust location estimate as fixed-point weighted mean with weights 1/||x - mu||^2."""
    n, p = X.shape
    mu = X.mean(axis=0)
    for _ in range(max_iter):
        d2 = np.sum((X - mu) ** 2, axis=1)
        d2 = np.maximum(d2, 1e-12)
        w = 1.0 / d2
        w /= w.sum()
        mu_new = (w[:, None] * X).sum(axis=0)
        if la.norm(mu_new - mu) < tol:
            break
        mu = mu_new
    return mu


def tylers_M(X, tol=1e-6, max_iter=500, assume_centered=True):
    n, p = X.shape
    Xc = X if assume_centered else (X - X.mean(0))
    C = np.cov(Xc, rowvar=False, bias=False)
    C = C / np.trace(C) * p
    for _ in range(max_iter):
        C_old = C
        invC = la.inv(C)
        denom = np.einsum("ij,jk,ik->i", Xc, invC, Xc)
        denom = np.maximum(denom, 1e-12)
        C = (p / n) * (
            Xc[:, :, None] * Xc[:, None, :] / denom[:, None, None]
        ).sum(axis=0)
        C = C / np.trace(C) * p
        if la.norm(C - C_old, "fro") < tol:
            break
    return C


def _mu_sigma2(c1, c2):
    # Theorem 2.4: limiting mean and variance for p*T2_tr
    mu = (
        -3 * c1
        + c1**2
        - 3 * c2
        + 8 * c1 * c2
        - 3 * (c1**2) * c2
        + c2**2
        - c1 * (c2**2)
    ) / ((c2 - 1) * (c1 - 1) ** 2)

    sigma2 = (
        4 * (c1**2)
        + 8 * (c1**3)
        + 8 * c1 * c2
        - 8 * (c1**3) * c2
        + 4 * (c2**2)
        - 8 * c1 * (c2**2)
        + 4 * (c1**2) * (c2**2)
    ) / (
        (1 - c1) ** 4
    )  # same as (c1-1)**4
    return mu, sigma2
