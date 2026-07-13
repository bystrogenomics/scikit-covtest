import numpy as np


def noise_reduction_pca(X, rank=1):
    """
    Perform noise-reduction PCA for HDLSS data using the correct formula from Ishii et al. (2015).

    Parameters
    ----------
    X : np.ndarray, shape (d, n)
        Data matrix with d features and n samples.
    rank : int
        Number of leading principal components to estimate.

    Returns
    -------
    lambda_tilde : np.ndarray
        Noise-reduced eigenvalue estimates of shape (rank,).
    h_tilde : np.ndarray
        Noise-reduced eigenvector estimates of shape (d, rank).
    s_tilde : np.ndarray
        Noise-reduced PC scores of shape (rank, n).
    """
    d, n = X.shape
    Xc = X - X.mean(axis=1, keepdims=True)
    S_dual = (Xc.T @ Xc) / (n - 1)
    eigvals, eigvecs = np.linalg.eigh(S_dual)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    trace_S = np.trace(S_dual)
    lambda_tilde = np.zeros(rank)
    for i in range(rank):
        residual = trace_S - np.sum(eigvals[: i + 1])
        lambda_tilde[i] = eigvals[i] - residual / (
            n - 2
        )  # Corrected denominator

    lambda_tilde = np.maximum(lambda_tilde, 0)

    h_tilde = []
    s_tilde = []
    for i in range(rank):
        u_i = eigvecs[:, i]
        h_i = Xc @ u_i / np.sqrt((n - 1) * lambda_tilde[i])
        s_i = np.sqrt((n - 1) * lambda_tilde[i]) * u_i
        h_tilde.append(h_i)
        s_tilde.append(s_i)

    return lambda_tilde, np.column_stack(h_tilde), np.vstack(s_tilde)
