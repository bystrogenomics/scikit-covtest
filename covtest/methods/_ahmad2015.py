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


def _standardize_T(T: float, n: int, p: int, calibration: str = "auto"):
    """
    Standardize T into a z-score under the null.

    Two asymptotic calibrations from the two papers:

    1) "large_p_small_n": (n/2) * T -> N(0, 1)  (equivalently nT -> N(0, 4))
       This is the regime emphasized in the non-normality note and also the c -> inf limit.

    2) "ratio": nT -> N(0, 4*(1 + 2/c)) where c = p/n  (JSCS paper; p/n -> c in (0, inf))

    "auto": use "large_p_small_n" if p > n, else "ratio".
    """
    if calibration not in {"auto", "large_p_small_n", "ratio"}:
        raise ValueError(
            "calibration must be one of: auto, large_p_small_n, ratio"
        )

    if calibration == "auto":
        calibration = "large_p_small_n" if p > n else "ratio"

    if calibration == "large_p_small_n":
        z = (n / 2.0) * T
        return z, calibration

    # ratio calibration
    c = p / n
    var_nT = 4.0 * (1.0 + 2.0 / c)  # = 4*(2/c + 1)
    z = (n * T) / np.sqrt(var_nT)
    return z, calibration
