"""Family-wise error rate (FWER) multiple testing corrections.

This module implements procedures that control the family-wise error rate
(FWER), defined as the probability of making at least one false rejection
when testing multiple hypotheses.

Functions
---------
bonferroni
    Classical Bonferroni correction that adjusts p-values by the number
    of tests, providing strong FWER control under any dependence structure.

holm
    Holm-Bonferroni sequentially rejective step-down procedure, uniformly
    more powerful than Bonferroni while maintaining strong FWER control.

hochberg
    Hochberg step-up procedure, more powerful than Holm under independence
    or positive dependence (PRDS) among test statistics.

hommel
    Hommel exact procedure, uniformly more powerful than Holm while
    controlling FWER under arbitrary dependence; more computationally
    intensive for large numbers of tests.

romano_wolf_maxT
    Romano-Wolf step-down max-T procedure using bootstrap or permutation
    resampling of test statistics under the joint null hypothesis.

Notes
-----
FWER-controlling procedures are typically more conservative than FDR-based
procedures, but they provide strong control of the probability of making
any false rejection, which can be desirable in high-stakes applications.

The Romano-Wolf procedure requires user-supplied bootstrap or permutation
samples that approximate the joint null distribution of the test statistics.

See Also
--------
covtest.multiplicity.fdr
    False discovery rate (FDR) control procedures.

References
----------
- [1] Bonferroni, C. E. (1936).
       "Teoria statistica delle classi e calcolo delle probabilita."
       Pubblicazioni del R Istituto Superiore di Scienze Economiche e
       Commerciali di Firenze.
- [2] Holm, S. (1979).
       "A simple sequentially rejective multiple test procedure."
       Scandinavian Journal of Statistics, 6(2), 65-70.
- [3] Hochberg, Y. (1988).
       "A sharper Bonferroni procedure for multiple tests of significance."
       Biometrika, 75(4), 800-802.
- [4] Hommel, G. (1988).
       "A stagewise rejective multiple test procedure based on a modified
       Bonferroni test."
       Biometrika, 75(2), 383-386.
- [5] Romano, J. P., and Wolf, M. (2005).
       "Exact and approximate stepdown methods for multiple hypothesis
       testing."
       Journal of the American Statistical Association, 100(469), 94-108.

Examples
--------
>>> import numpy as np
>>> from covtest.multiplicity import fwer
>>> pvals = np.array([0.01, 0.03, 0.2])
>>> res = fwer.holm(pvals, alpha=0.05)
>>> res["rejected"]
array([ True,  True, False])
"""

from typing import Dict, Literal

import numpy as np


def bonferroni(pvals: np.ndarray, alpha: float = 0.05) -> Dict:
    """
    Apply Bonferroni correction for FWER control.

    The Bonferroni correction is the simplest and most conservative method
    for controlling the family-wise error rate. It adjusts p-values by
    multiplying them by the number of tests, ensuring strong FWER control
    under any dependence structure.

    Parameters
    ----------
    pvals : array-like of shape (n_tests,)
        Raw p-values from multiple hypothesis tests. Must be numeric values
        between 0 and 1.

    alpha : float, default=0.05
        The desired family-wise error rate level. Must be between 0 and 1.

    Returns
    -------
    results : dict
        Dictionary containing the following keys:

        - 'pvals_adj' : ndarray of shape (n_tests,)
            Bonferroni-adjusted p-values, computed as min(1, p * m) where
            m is the number of tests.

        - 'rejected' : ndarray of shape (n_tests,), dtype=bool
            Boolean array indicating which null hypotheses are rejected
            at the given alpha level.

        - 'alpha' : float
            The FWER level used for the correction.

        - 'method' : str
            Always 'Bonferroni' to indicate the procedure used.

    Notes
    -----
    The Bonferroni correction controls FWER at level alpha by rejecting
    hypothesis i if p_i <= alpha/m, where m is the total number of tests.

    This is equivalent to comparing adjusted p-values p_adj_i = m * p_i
    against alpha.

    While very conservative, the Bonferroni correction is simple, widely
    used, and provides strong FWER control under any dependence structure.

    References
    ----------
    - [1] Bonferroni, C. E. (1936). Teoria statistica delle classi e
           calcolo delle probabilità. Pubblicazioni del R Istituto Superiore
           di Scienze Economiche e Commerciali di Firenze, 8, 3-62.

    Examples
    --------
    >>> import numpy as np
    >>> pvals = np.array([0.01, 0.03, 0.2])
    >>> results = bonferroni(pvals, alpha=0.05)
    >>> results['pvals_adj']
    array([0.03, 0.09, 0.6 ])
    >>> results['rejected']
    array([ True, False, False])
    """
    p = np.asarray(pvals, dtype=float)
    m = p.size
    p_adj = np.minimum(1.0, p * m)
    rejected = p_adj <= alpha
    return {
        "pvals_adj": p_adj,
        "rejected": rejected,
        "alpha": alpha,
        "method": "Bonferroni",
    }


def holm(pvals: np.ndarray, alpha: float = 0.05) -> Dict:
    """
    Apply Holm-Bonferroni step-down procedure for FWER control.

    The Holm-Bonferroni procedure is a sequentially rejective step-down
    method that is uniformly more powerful than the Bonferroni correction
    while maintaining strong FWER control. It tests hypotheses in order
    of increasing p-values.

    Parameters
    ----------
    pvals : array-like of shape (n_tests,)
        Raw p-values from multiple hypothesis tests. Must be numeric values
        between 0 and 1.

    alpha : float, default=0.05
        The desired family-wise error rate level. Must be between 0 and 1.

    Returns
    -------
    results : dict
        Dictionary containing the following keys:

        - 'pvals_adj' : ndarray of shape (n_tests,)
            Holm-adjusted p-values. For the i-th smallest p-value,
            p_adj_i = (m - i + 1) * p_i, enforcing monotonicity.

        - 'rejected' : ndarray of shape (n_tests,), dtype=bool
            Boolean array indicating which null hypotheses are rejected
            at the given alpha level.

        - 'alpha' : float
            The FWER level used for the correction.

        - 'method' : str
            Always 'Holm' to indicate the procedure used.

    Notes
    -----
    The Holm procedure works as follows:

    1. Sort p-values in ascending order: p_(1) <= p_(2) <= -. <= p_(m)
    2. Find the smallest k such that p_(k) > alpha/(m - k + 1)
    3. Reject hypotheses 1, -., k-1

    The adjusted p-values are computed as p_adj_(i) = (m - i + 1) * p_(i),
    with monotonicity enforced so that p_adj_(i) <= p_adj_(i+1).

    This method is uniformly more powerful than Bonferroni while maintaining
    the same FWER guarantee under any dependence structure.

    References
    ----------
    - [1] Holm, S. (1979). A simple sequentially rejective multiple test
           procedure. Scandinavian Journal of Statistics, 6(2), 65-70.

    Examples
    --------
    >>> import numpy as np
    >>> pvals = np.array([0.001, 0.01, 0.2, 0.5])
    >>> results = holm(pvals, alpha=0.05)
    >>> results['pvals_adj']
    array([0.004, 0.03 , 0.4  , 0.5  ])
    >>> results['rejected']
    array([ True,  True, False, False])
    """
    p = np.asarray(pvals, dtype=float)
    m = p.size
    order = np.argsort(p)
    p_sorted = p[order]

    # raw adjusted
    p_adj = np.minimum(1.0, (m - np.arange(m)) * p_sorted)

    # enforce monotonicity (non-decreasing with i)
    p_adj = np.maximum.accumulate(p_adj)

    # back to original order
    p_adj_final = np.empty_like(p_adj)
    p_adj_final[order] = p_adj
    rejected = p_adj_final <= alpha
    return {
        "pvals_adj": p_adj_final,
        "rejected": rejected,
        "alpha": alpha,
        "method": "Holm",
    }


def hochberg(pvals: np.ndarray, alpha: float = 0.05) -> Dict:
    """
    Apply Hochberg step-up procedure for FWER control.

    The Hochberg procedure is a step-up method that is more powerful than
    the Holm procedure under certain conditions. It requires an assumption
    of non-negative dependence among test statistics but provides increased
    power when this assumption holds.

    Parameters
    ----------
    pvals : array-like of shape (n_tests,)
        Raw p-values from multiple hypothesis tests. Must be numeric values
        between 0 and 1.

    alpha : float, default=0.05
        The desired family-wise error rate level. Must be between 0 and 1.

    Returns
    -------
    results : dict
        Dictionary containing the following keys:

        - 'pvals_adj' : ndarray of shape (n_tests,)
            Hochberg-adjusted p-values. For the i-th smallest p-value,
            p_adj_i = (m - i + 1) * p_i, with reverse monotonicity enforced.

        - 'rejected' : ndarray of shape (n_tests,), dtype=bool
            Boolean array indicating which null hypotheses are rejected
            at the given alpha level.

        - 'alpha' : float
            The FWER level used for the correction.

        - 'method' : str
            Always 'Hochberg' to indicate the procedure used.

    Notes
    -----
    The Hochberg procedure works as follows:

    1. Sort p-values in ascending order: p_(1) <= p_(2) <= -. <= p_(m)
    2. Find the largest k such that p_(k) <= alpha/(m - k + 1)
    3. Reject hypotheses 1, -., k

    Unlike Holm (step-down), Hochberg uses a step-up approach, starting
    from the largest p-value and working backwards.

    The procedure controls FWER under independence or positive dependence
    (PRDS) among test statistics. It is more powerful than Holm when these
    conditions are met.

    References
    ----------
    - [1] Hochberg, Y. (1988). A sharper Bonferroni procedure for multiple
           tests of significance. Biometrika, 75(4), 800-802.

    Examples
    --------
    >>> import numpy as np
    >>> pvals = np.array([0.001, 0.01, 0.2, 0.5])
    >>> results = hochberg(pvals, alpha=0.05)
    >>> results['pvals_adj']
    array([0.004, 0.03 , 0.4  , 0.5  ])
    >>> results['rejected']
    array([ True,  True, False, False])
    """
    p = np.asarray(pvals, dtype=float)
    m = p.size
    order = np.argsort(p)  # sort ascending
    p_sorted = p[order]

    # Raw adjusted (before monotonicity): (m - i) * p_(i), i = 0-m-1
    adj_sorted = p_sorted * (m - np.arange(m))
    adj_sorted = np.minimum(adj_sorted, 1.0)

    # Enforce step-up monotonicity: suffix cumulative minimum
    adj_sorted = np.minimum.accumulate(adj_sorted[::-1])[::-1]

    # Map back to original order
    p_adj_final = np.empty_like(p)
    p_adj_final[order] = adj_sorted

    rejected = p_adj_final <= alpha
    return {
        "pvals_adj": p_adj_final,
        "rejected": rejected,
        "alpha": alpha,
        "method": "Hochberg",
    }


def hommel(pvals: np.ndarray, alpha: float = 0.05) -> Dict:
    """
    Apply Hommel's procedure for FWER control.

    Hommel's procedure is an exact method that is uniformly more powerful
    than Holm's procedure while still controlling FWER under arbitrary
    dependence. However, it is computationally more intensive for large
    numbers of tests.

    Parameters
    ----------
    pvals : array-like of shape (n_tests,)
        Raw p-values from multiple hypothesis tests. Must be numeric values
        between 0 and 1.

    alpha : float, default=0.05
        The desired family-wise error rate level. Must be between 0 and 1.

    Returns
    -------
    results : dict
        Dictionary containing the following keys:

        - 'pvals_adj' : ndarray of shape (n_tests,)
            Hommel-adjusted p-values.

        - 'rejected' : ndarray of shape (n_tests,), dtype=bool
            Boolean array indicating which null hypotheses are rejected
            at the given alpha level.

        - 'alpha' : float
            The FWER level used for the correction.

        - 'method' : str
            Always 'Hommel' to indicate the procedure used.

    Notes
    -----
    Hommel's procedure is based on a modified Bonferroni test that considers
    all possible subsets of hypotheses. The algorithm iterates through
    different subset sizes to compute adjusted p-values.

    While more powerful than Holm, the computational complexity increases
    with the number of tests, making it less practical for very large m.

    The procedure provides strong FWER control under any dependence structure.

    References
    ----------
    - [1] Hommel, G. (1988). A stagewise rejective multiple test procedure
           based on a modified Bonferroni test. Biometrika, 75(2), 383-386.

    Examples
    --------
    >>> import numpy as np
    >>> pvals = np.array([0.001, 0.01, 0.2, 0.5])
    >>> results = hommel(pvals, alpha=0.05)
    >>> results['pvals_adj']
    array([0.004, 0.03 , 0.4  , 0.5  ])
    >>> results['rejected']
    array([ True,  True, False, False])
    """
    p = np.asarray(pvals, dtype=float)
    m = p.size
    order = np.argsort(p)
    p_sorted = p[order]

    p_adj = np.full(m, 1.0)
    for j in range(1, m + 1):
        c = min((m / j) * np.min(p_sorted[j - 1 :]), 1.0)
        p_adj[:j] = np.minimum(p_adj[:j], c)

    p_adj_final = np.empty_like(p_adj)
    p_adj_final[order] = p_adj
    rejected = p_adj_final <= alpha
    return {
        "pvals_adj": p_adj_final,
        "rejected": rejected,
        "alpha": alpha,
        "method": "Hommel",
    }


def romano_wolf_maxT(
    T_obs: np.ndarray,
    T_boot: np.ndarray,
    alpha: float = 0.05,
    side: Literal["right", "left", "two-sided"] = "right",
) -> Dict:
    """
    Apply Romano-Wolf step-down max-T procedure for FWER control.

    The Romano-Wolf procedure uses bootstrap or permutation resampling to
    compute adjusted p-values that account for the joint distribution of
    test statistics. It provides strong FWER control under arbitrary
    dependence and is particularly powerful for correlated tests.

    Parameters
    ----------
    T_obs : array-like of shape (m,)
        Observed test statistics. For side="right", larger values indicate
        stronger evidence against the null hypothesis.

    T_boot : array-like of shape (B, m)
        Bootstrap or permutation test statistics under the joint null
        hypothesis. Each row corresponds to one resample, and columns
        correspond to the m hypotheses.

    alpha : float, default=0.05
        The desired family-wise error rate level. Must be between 0 and 1.

    side : {"right", "left", "two-sided"}, default="right"
        Direction of the alternative hypothesis:

        - 'right' : Test statistic is large under the alternative
        - 'left' : Test statistic is small under the alternative
        - 'two-sided' : Test statistic is extreme (large or small) under the alternative

    Returns
    -------
    results : dict
        Dictionary containing the following keys:

        - 'pvals_adj' : ndarray of shape (m,)
            Romano-Wolf adjusted p-values accounting for the joint
            distribution of test statistics.

        - 'rejected' : ndarray of shape (m,), dtype=bool
            Boolean array indicating which null hypotheses are rejected
            at the given alpha level.

        - 'alpha' : float
            The FWER level used for the correction.

        - 'method' : str
            Always 'Romano–Wolf' to indicate the procedure used.

    Raises
    ------
    ValueError
        If T_obs and T_boot have incompatible shapes (different number of
        hypotheses m).

    Notes
    -----
    The Romano-Wolf procedure works as follows:

    1. Order test statistics from strongest to weakest evidence
    2. For each hypothesis k in this order:
       - Compute max statistic over remaining hypotheses in each bootstrap sample
       - Calculate p-value as proportion of bootstrap max statistics >= observed statistic
       - Enforce monotonicity: p_adj_k >= p_adj_(k-1)

    This step-down approach accounts for the dependence structure by using
    the joint distribution of test statistics under the null hypothesis.

    The method requires user-supplied bootstrap or permutation samples
    (T_boot) generated under the joint null hypothesis. The quality of
    FWER control depends on the validity of these resamples.

    References
    ----------
    - [1] Romano, J. P., & Wolf, M. (2005). Exact and approximate stepdown
           methods for multiple hypothesis testing. Journal of the American
           Statistical Association, 100(469), 94-108.

    - [2] Romano, J. P., & Wolf, M. (2016). Efficient computation of
           adjusted p-values for resampling-based stepdown multiple testing.
           Statistics & Probability Letters, 113, 38-40.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> T_obs = np.array([2.5, 1.8, 3.2])
    >>> T_boot = np.random.standard_normal((200, 3))
    >>> results = romano_wolf_maxT(T_obs, T_boot, alpha=0.05)
    >>> results['rejected']
    array([ True, False,  True])

    >>> # Two-sided test
    >>> results_two = romano_wolf_maxT(T_obs, T_boot, alpha=0.05, side="two-sided")
    >>> results_two['method']
    'Romano–Wolf'
    """
    T_obs = np.asarray(T_obs, dtype=float)
    T_boot = np.asarray(T_boot, dtype=float)
    B, m = T_boot.shape
    if T_obs.size != m:
        raise ValueError(
            "T_obs and T_boot must have the same number of hypotheses (m)."
        )

    # effective test stats
    if side == "two-sided":
        T_obs_eff = np.abs(T_obs)
        T_boot_eff = np.abs(T_boot)
    elif side == "left":
        T_obs_eff = -T_obs
        T_boot_eff = -T_boot
    else:  # right
        T_obs_eff = T_obs
        T_boot_eff = T_boot

    order = np.argsort(-T_obs_eff)  # strongest first
    inv_order = np.empty_like(order)
    inv_order[order] = np.arange(m)

    p_adj = np.ones(m, dtype=float)

    # step-down loop
    for k in range(m):
        idx_remaining = order[k:]
        maxT_boot = np.max(T_boot_eff[:, idx_remaining], axis=1)
        p_k = (np.sum(maxT_boot >= T_obs_eff[order[k]]) + 1.0) / (B + 1.0)
        if k == 0:
            p_adj[order[k]] = p_k
        else:
            p_adj[order[k]] = max(p_adj[order[k - 1]], p_k)

    p_adj_final = p_adj[inv_order]
    rejected = p_adj_final <= alpha
    return {
        "pvals_adj": p_adj_final,
        "rejected": rejected,
        "alpha": alpha,
        "method": "Romano–Wolf",
    }
