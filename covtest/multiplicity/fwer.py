"""
Family-Wise Error Rate (FWER) control procedures.

This module implements classical and modern procedures for controlling
the family-wise error rate, i.e. the probability of making one or more
false rejections when testing multiple hypotheses.

Implemented methods
-------------------
- Bonferroni (1936)
- Holm–Bonferroni step-down (1979)
- Hochberg step-up (1988)
- Hommel exact procedure (1988)
- Romano–Wolf step-down max-T (2005) with bootstrap/permutation resampling

References
----------
.. [1] Bonferroni, C. E. (1936). "Teoria statistica delle classi e calcolo delle probabilità".
       Pubblicazioni del R Istituto Superiore di Scienze Economiche e Commerciali di Firenze.
.. [2] Holm, S. (1979). "A simple sequentially rejective multiple test procedure".
       Scandinavian Journal of Statistics, 6(2), 65–70.
.. [3] Hochberg, Y. (1988). "A sharper Bonferroni procedure for multiple tests of significance".
       Biometrika, 75(4), 800–802.
.. [4] Hommel, G. (1988). "A stagewise rejective multiple test procedure 
         based on a modified Bonferroni test".
       Biometrika, 75(2), 383–386.
.. [5] Romano, J. P., & Wolf, M. (2005). "Exact and approximate stepdown 
       methods for multiple hypothesis testing".
       Journal of the American Statistical Association, 100(469), 94–108.

Notes
-----
The Romano–Wolf procedure requires user-supplied bootstrap or permutation
test statistics under the joint null hypothesis. It provides strong control
of FWER under arbitrary dependence.

Examples
--------
>>> import numpy as np
>>> from covtest.multiplicity import fwer
>>> pvals = np.array([0.01, 0.03, 0.2])
>>> res = fwer.holm(pvals, alpha=0.05)
>>> res["rejected"]
array([ True,  True, False])
"""
__all__ = ["bonferroni", "holm", "hochberg", "hommel", "romano_wolf_maxT"]


from typing import Dict, Literal

import numpy as np


def bonferroni(pvals: np.ndarray, alpha: float = 0.05) -> Dict:
    """Bonferroni correction for FWER control."""
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
    """Holm–Bonferroni step-down procedure for FWER control."""
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
    """Hochberg step-up procedure for FWER control."""
    p = np.asarray(pvals, dtype=float)
    m = p.size
    order = np.argsort(p)  # sort ascending
    p_sorted = p[order]

    # Raw adjusted (before monotonicity): (m - i) * p_(i), i = 0..m-1
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
    """Hommel’s procedure for FWER control (exact, but less powerful for large m)."""
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
    Romano–Wolf step-down max-T adjusted p-values for FWER control.

    Parameters
    ----------
    T_obs : array-like of shape (m,)
        Observed test statistics (large = more evidence if side="right").
    T_boot : array-like of shape (B, m)
        Bootstrap/permutation test statistics under the joint null.
        Each row corresponds to one resample, columns to hypotheses.
    alpha : float, default=0.05
        Target FWER level.
    side : {"right", "left", "two-sided"}, default="right"
        Alternative direction.

    Returns
    -------
    results : dict
        - 'pvals_adj': adjusted p-values
        - 'rejected': boolean array of rejections at level alpha
        - 'alpha': significance level
        - 'method': string identifier
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
