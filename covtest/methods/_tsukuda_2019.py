import numpy as np


def _unbiased_tr_sigma2_per_p(X: np.ndarray) -> float:
    """
    Unbiased estimator of tr(Sigma^2)/p for i.i.d. vectors with finite 8th moments.
    Uses the closed form in Srivastava–Yanagihara–Kubokawa (2014), Eq. (2.5):
        a2_hat = [ (N-2)*n*tr(V^2) - N*n*tr(D^2) + (tr V)^2 ] / [ p*N*(N-1)*(N-2)*(N-3) ]
    where:
      - X is shape (N, p)
      - y_j = x_j - mean(x)
      - V = sum_j y_j y_j^T  (p×p scatter)
      - D = diag( ||y_1||^2, ..., ||y_N||^2 ) (N×N)
      - n = N - 1
    Returns tr(Sigma^2)/p estimate.
    References: CIRJE-F-933 (June 2014), Sec. 2, Eq. (2.5).
    """
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array of shape (N, p).")
    N, p = X.shape
    if N < 4:
        raise ValueError("Unbiased tr(Sigma^2) requires N >= 4.")
    Y = X - X.mean(axis=0, keepdims=True)  # (N, p)
    V = Y.T @ Y  # (p, p)
    s = np.einsum("ij,ij->i", Y, Y)  # ||y_j||^2 for j=1..N
    tr_V2 = np.sum(V * V)  # tr(V^2)
    tr_D2 = np.sum(s * s)  # tr(D^2)
    tr_V = np.sum(s)  # tr(V)
    n = N - 1
    denom = p * N * (N - 1) * (N - 2) * (N - 3)
    a2_hat = ((N - 2) * n * tr_V2 - N * n * tr_D2 + tr_V ** 2) / denom
    return a2_hat


def _a2_hat_tsukuda(X: np.ndarray) -> float:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    N, p = X.shape
    if N < 3:
        raise ValueError("Tsukuda-Matsuura a2 estimator requires N >= 3.")

    Xc = X - X.mean(axis=0, keepdims=True)
    S = (Xc.T @ Xc) / (N - 1)

    trS = np.trace(S)
    trS2 = np.sum(S * S)

    return ((N - 1) ** 2 / (p * (N - 2) * (N + 1))) * (trS2 - (trS ** 2) / (N - 1))
