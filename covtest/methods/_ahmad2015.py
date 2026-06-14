import numpy as np


def _trace_estimators_from_gram(G: np.ndarray):
    """
    Given Gram matrix G = X X^T (n x n), compute the U-statistic trace estimators:

    E1 = (1/n) sum_k A_k               estimates tr(Sigma)
    E2 = (1/(n(n-1))) sum_{k!=l} A_k A_l  estimates (tr(Sigma))^2
    E3 = (1/(n(n-1))) sum_{k!=l} (A_kl)^2  estimates tr(Sigma^2)

    where A_k = G_kk and A_kl = G_kl.
    """
    n = G.shape[0]
    if G.shape[1] != n:
        raise ValueError("G must be square (n x n).")
    if n < 2:
        raise ValueError("Need n >= 2 samples.")

    diag = np.diag(G)
    sum_diag = float(np.sum(diag))
    sum_diag2 = float(np.sum(diag * diag))

    # E1
    E1 = sum_diag / n

    # sum_{k!=l} A_k A_l = (sum_k A_k)^2 - sum_k A_k^2
    E2_num = (sum_diag * sum_diag) - sum_diag2
    E2 = E2_num / (n * (n - 1))

    # sum_{k!=l} (A_kl)^2 = sum_{k,l} G_kl^2 - sum_k G_kk^2
    G2_sum = float(np.sum(G * G))
    E3_num = G2_sum - sum_diag2
    E3 = E3_num / (n * (n - 1))

    return E1, E2, E3


def _standardize_T(T: float, n: int, p: int, calibration: str = "ahmad2015"):
    """
    Standardize an Ahmad/von Rosen (2015) T statistic under the null.

    For both the Ahmad/von Rosen 2015 sphericity statistic T1 and
    identity statistic T2, the null limit is:

        (n / 2) * T -> N(0, 1)

    Therefore the default z-score is:

        z = (n / 2) * T

    The old "large_p_small_n" name is retained as an alias. The "ratio"
    calibration is explicit opt-in only, is not the Ahmad/von Rosen 2015
    calibration, and is not selected by "auto".
    """
    allowed = {"auto", "ahmad2015", "large_p_small_n", "ratio"}
    if calibration not in allowed:
        raise ValueError(
            "calibration must be one of: auto, ahmad2015, large_p_small_n, ratio"
        )

    if calibration in {"auto", "ahmad2015", "large_p_small_n"}:
        return (n / 2.0) * T, "ahmad2015"

    c = p / n
    var_nT = 4.0 * (1.0 + 2.0 / c)
    return (n * T) / np.sqrt(var_nT), "ratio"
