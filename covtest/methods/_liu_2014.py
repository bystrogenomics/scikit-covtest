import numpy as np


def _center_rows(M):
    # center columns (variables): shape (N, p)
    return M - M.mean(axis=0, keepdims=True)


def _unbiased_cov(M):
    # M: (N, p) data matrix, rows = observations
    N = M.shape[0]
    Xc = _center_rows(M)
    return (Xc.T @ Xc) / (N - 1)


def _beta_hat(data):
    """
    Consistent estimator of excess kurtosis parameter beta (Theorem 3.2).
    data: (N, p), rows are observations; will be centered inside.
    """
    X = _center_rows(data)
    N, p = X.shape

    # norms and traces
    norms2 = np.einsum("ij,ij->i", X, X)  # ||X_i||^2
    sum_norms2 = norms2.sum()
    sum_norms4 = (norms2 ** 2).sum()

    # sum_{i != j} tr(X_i X_i^T X_j X_j^T) = sum_{i != j} (X_i^T X_j)^2
    G = X @ X.T  # Gram matrix
    off_diag_sq_sum = (G ** 2).sum() - np.diag(G ** 2).sum()

    num = (
        sum_norms4 / (p * (N - 1))
        - (sum_norms2 ** 2) / (p * N * (N - 1))
        - 2 * off_diag_sq_sum / (p * N * (N - 1))
    )

    denom = (1.0 / p) * (
        (X ** 2).mean(axis=0) ** 2
    ).sum()  # (1/p) * sum_k ( (1/N) sum_i x_{ik}^2 )^2

    return num / denom
