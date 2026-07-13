from typing import List, Optional, Union

import numpy as np
from scipy.linalg import eigh  # type: ignore
from scipy.stats import norm


def two_way_sampling(X: np.ndarray, Y: np.ndarray, n: int, seed: int = 2021) -> dict:
    """
    Perform two-way sampling on matrices X and Y.

    Parameters
    ----------
    X : np.ndarray
        First matrix.
    Y : np.ndarray
        Second matrix.
    n : int
        Sample size.
    seed : int, optional
        Random seed, by default 2021.

    Returns
    -------
    sampled_matrices : dict
        Dictionary containing sampled matrices Xs, Ys, Zs.
    """
    rng = np.random.default_rng(seed)
    n1, n2 = X.shape[0], Y.shape[0]
    if n > min(max(n1, n2) / 2, n1, n2):
        raise ValueError("Invalid sample sizes!")

    if n1 <= n2:
        xs_idx = rng.choice(n1, n, replace=False)
        ys_idx = rng.choice(n2, 2 * n, replace=False)
        zs_idx = ys_idx[n:]
        ys_idx = ys_idx[:n]
        Xs = X[xs_idx, :]
        Ys = Y[ys_idx, :]
        Zs = Y[zs_idx, :]
    else:
        xs_idx = rng.choice(n1, 2 * n, replace=False)
        ys_idx = rng.choice(n2, n, replace=False)
        zs_idx = xs_idx[n:]
        xs_idx = xs_idx[:n]
        Xs = X[xs_idx, :]
        Ys = Y[ys_idx, :]
        Zs = X[zs_idx, :]

    return {"Xs": Xs, "Ys": Ys, "Zs": Zs}


def cov_eigs(X: np.ndarray) -> np.ndarray:
    """
    Find eigenvalues of the sample covariance matrix.

    Parameters
    ----------
    X : np.ndarray
        Input matrix.

    Returns
    -------
    eigvals : np.ndarray
        Eigenvalues of the sample covariance matrix.
    """
    n, p = X.shape
    X1 = X - np.mean(X, axis=0)
    return eigh(np.dot(X1, X1.T) / np.sqrt(n * p), eigvals_only=True)


def k_func(x: float) -> float:
    """
    Kernel function K.

    Parameters
    ----------
    x : float
        Input value.

    Returns
    -------
    k_value : float
        Output of the kernel function.
    """
    if abs(x) >= 1.05:
        return 0
    if abs(x) <= 1:
        return 1
    return np.exp(1 / 0.05 ** 2 - 1 / (0.05 ** 2 - (abs(x) - 1) ** 2))


def t_func(lambda_: np.ndarray, gamma: float, eta0: float) -> float:
    """
    Compute T value.

    Parameters
    ----------
    lambda_ : np.ndarray
        Array of eigenvalues.
    gamma : float
        Gamma value.
    eta0 : float
        Eta value.

    Returns
    -------
    t_value : float
        T value.
    """
    return sum(
        (lambda_i - gamma) / eta0 * k_func((lambda_i - gamma) / eta0)
        for lambda_i in lambda_
    )


def check_efficient(
    gamma: float, lambda1: np.ndarray, lambda2: np.ndarray, epsilon: float = 0.05,
) -> bool:
    """
    Check if the splitting is efficient.
    Parameters
    ----------
    gamma : float
        Gamma value.
    lambda1 : np.ndarray
        First array of eigenvalues.
    lambda2 : np.ndarray
        Second array of eigenvalues.
    epsilon : float, optional
        Tolerance value, by default 0.05.
    Returns
    -------
    is_efficient : bool
        True if efficient, False otherwise.
    """
    range1 = abs(lambda1[0] - lambda1[-1])
    range2 = abs(lambda2[0] - lambda2[-1])
    return not (
        max(abs(gamma - lambda1[0]), abs(gamma - lambda1[-1])) > range1 - epsilon
        or max(abs(gamma - lambda2[0]), abs(gamma - lambda2[-1])) > range2 - epsilon
    )


def two_sample_test_(
    X: np.ndarray,
    Y: np.ndarray,
    n: int,
    const: float = 0.5,
    alpha: float = 0.05,
    epsilon: float = 0.05,
    thres: Optional[float] = None,
    mode: str = "test",
    seed: int = 2021,
) -> Union[dict, float]:
    """
    Perform a two-sample test.

    Parameters
    ----------
    X : np.ndarray
        First matrix.
    Y : np.ndarray
        Second matrix.
    n : int
        Sample size.
    const : float, optional
        Constant value, by default 0.5.
    alpha : float, optional
        Significance level, by default 0.05.
    epsilon : float, optional
        Tolerance value, by default 0.05.
    thres : Optional[float], optional
        Threshold value, by default None.
    mode : str, optional
        Mode of the test, by default "test".
    seed : int, optional
        Random seed, by default 2021.

    Returns
    -------
    test_result : dict
        Result of the two-sample test.
    """
    sample_list = two_way_sampling(X, Y, n, seed=seed)
    Xs, Ys, Zs = sample_list["Xs"], sample_list["Ys"], sample_list["Zs"]

    eig_xs = cov_eigs(Xs)
    eig_ys = cov_eigs(Ys)
    eig_zs = cov_eigs(Zs)
    gamma = np.median(eig_zs)
    if not check_efficient(gamma, eig_xs, eig_ys, epsilon):
        return {"efficient": False, "c": 1}

    eta0 = np.std(eig_zs, ddof=1) * const

    Tx = t_func(eig_xs, gamma, eta0)
    Ty = t_func(eig_ys, gamma, eta0)

    if mode != "test":
        return abs(Tx - Ty)

    if thres is None:
        thres = 2.6

    if abs(Tx - Ty) > (thres / norm.ppf(1 - 0.05 / 2) * norm.ppf(1 - alpha / 2)):
        return {
            "efficient": True,
            "c": 1,
            "statistic": (Tx - Ty) / (thres / norm.ppf(1 - 0.05 / 2)),
        }
    return {
        "efficient": True,
        "c": 0,
        "statistic": (Tx - Ty) / (thres / norm.ppf(1 - 0.05 / 2)),
    }


def calibration(
    n1: int,
    n2: int,
    p: int,
    n: int,
    alpha: float = 0.05,
    const: float = 0.5,
    iterations: int = 100,
    K: int = 100,
    seed: int = 2021,
) -> float:
    """
    Perform calibration to find threshold.

    Parameters
    ----------
    n1 : int
        Size of the first sample.
    n2 : int
        Size of the second sample.
    p : int
        Number of columns.
    n : int
        Sample size.
    alpha : float, optional
        Significance level, by default 0.05.
    const : float, optional
        Constant value, by default 0.5.
    iterations : int, optional
        Number of iterations, by default 100.
    K : int, optional
        Number of sub-iterations, by default 100.
    seed : int, optional
        Random seed, by default 2021.

    Returns
    -------
    threshold : float
        Threshold value.
    """
    rng = np.random.default_rng(seed)
    values: List[float] = []
    for _ in range(iterations):
        X = rng.normal(0, 1, (n1, p))
        Y = rng.normal(0, 1, (n2, p))
        for _ in range(K):
            value = two_sample_test_(
                X,
                Y,
                n,
                alpha=alpha,
                const=const,
                mode="calib",
                seed=rng.integers(int(1e6)),
            )
            if isinstance(value, float):
                values.append(value)
    return float(np.quantile(values, 1 - alpha))


def c_tuning(
    X: np.ndarray,
    Y: np.ndarray,
    n: int,
    thres: Optional[float] = None,
    alpha: float = 0.05,
    epsilon: float = 0.05,
    K: int = 500,
    seed: int = 2021,
) -> dict:
    """
    Tune the constant C.

    Parameters
    ----------
    X : np.ndarray
        First matrix.
    Y : np.ndarray
        Second matrix.
    n : int
        Sample size.
    thres : Optional[float], optional
        Threshold value, by default None.
    alpha : float, optional
        Significance level, by default 0.05.
    epsilon : float, optional
        Tolerance value, by default 0.05.
    K : int, optional
        Number of iterations, by default 500.
    seed : int, optional
        Random seed, by default 2021.

    Returns
    -------
    tuning_result : dict
        Tuned constant C and rates.
    """
    rng = np.random.default_rng(seed)
    Cs = np.arange(1, 31) / 10
    rates = []

    if thres is None:
        thres = 2.6

    for c in Cs:
        all_ = 0
        rej = 0
        for _ in range(K):
            result = two_sample_test_(
                X,
                Y,
                n,
                alpha=alpha,
                const=c,
                epsilon=epsilon,
                thres=thres,
                seed=rng.integers(int(1e6)),
            )
            if isinstance(result, dict) and result["efficient"]:
                all_ += 1
                rej += result["c"]
        rates.append(rej / all_ if all_ > 50 else 1)

    stable = find_stable(rates)
    return {"c": Cs[stable], "rates": rates}


def find_stable(xs: List[float]) -> int:
    """
    Find the stable point in a list of rates.

    Parameters
    ----------
    xs : list
        List of rates.

    Returns
    -------
    stable_index : int
        Index of the stable point.
    """
    roll_average = np.convolve(xs, np.ones(3) / 3, mode="valid")
    vars_ = [np.var(roll_average[:i]) for i in range(2, len(roll_average))]
    for i in range(len(vars_) - 1):
        if vars_[i + 1] < vars_[i] and roll_average[i] > max(roll_average) / 5:
            break
    return i + 2


def movevar(xs: List[float]) -> List[float]:
    """
    Compute the moving variance of a list.

    Parameters
    ----------
    xs : list
        List of values.

    Returns
    -------
    variances : list
        List of variances.
    """
    n = len(xs)
    vars_ = []
    for i in range(2, n + 1):
        vars_.append(float(np.var(xs[:i])))
    return vars_
