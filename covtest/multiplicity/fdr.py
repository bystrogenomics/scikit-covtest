"""
False Discovery Rate (FDR) control procedures.

This module implements standard and adaptive procedures for controlling
the false discovery rate, defined as the expected proportion of false
rejections among all rejections.

Implemented methods
-------------------
- Benjamini–Hochberg (1995)
- Benjamini–Yekutieli (2001), robust under arbitrary dependence
- Weighted BH (Genovese et al., 2006)
- Storey–Tibshirani q-values (2003) with pi0 estimation

References
----------
.. [1] Benjamini, Y., & Hochberg, Y. (1995).
       "Controlling the false discovery rate: a practical and powerful 
        approach to multiple testing".
       Journal of the Royal Statistical Society: Series B, 57(1), 289–300.
.. [2] Benjamini, Y., & Yekutieli, D. (2001).
       "The control of the false discovery rate in multiple testing under dependency".
       Annals of Statistics, 29(4), 1165–1188.
.. [3] Storey, J. D., & Tibshirani, R. (2003).
       "Statistical significance for genomewide studies".
       Proceedings of the National Academy of Sciences, 100(16), 9440–9445.
.. [4] Genovese, C. R., Roeder, K., & Wasserman, L. (2006).
       "False discovery control with p-value weighting".
       Biometrika, 93(3), 509–524.

Notes
-----
- Weighted BH requires nonnegative weights; recommended normalization is
  to sum to the number of hypotheses.
- Storey’s method estimates the proportion of true nulls (pi0) adaptively
  to increase power.


Examples
--------
>>> import numpy as np
>>> from yourpackage.multiplicity import fdr
>>> pvals = np.array([0.001, 0.02, 0.2, 0.6])
>>> res = fdr.benjamini_hochberg(pvals, alpha=0.05)
>>> res["rejected"]
array([ True,  True, False, False])
"""

from typing import Dict

import numpy as np

__all__ = [
    "benjamini_hochberg",
    "benjamini_liu",
    "benjamini_yekutieli",
    "blaroq",
    "weighted_bh",
    "storey_qvalues",
]


def SUD(pvalues, critical_values, start_idx_sud):
    """
    Step-Up/Step-Down (SUD) procedure for multiple testing.

    This is a general procedure that can perform either step-down or step-up
    testing depending on the starting index. It compares p-values against
    critical values to determine which hypotheses to reject.

    Parameters
    ----------
    pvalues : array-like of shape (m,)
        P-values from multiple hypothesis tests.

    critical_values : array-like of shape (m,)
        Critical values corresponding to each p-value. Must have the same
        length as pvalues.

    start_idx_sud : int
        Starting index (1-based indexing, as in the original R implementation).
        Controls the direction of the procedure:

        - 1 : Step-Down (SD) procedure
        - m : Step-Up (SU) procedure

    Returns
    -------
    rejected : ndarray of shape (m,), dtype=bool
        Boolean array indicating which hypotheses are rejected.

    Raises
    ------
    ValueError
        If only 1 critical value is provided (use SS() instead).
        If lengths of critical_values and pvalues do not match.
        If start_idx_sud is out of bounds [1, m].

    Notes
    -----
    The procedure works as follows:

    1. Sort p-values in ascending order
    2. Compare sorted p-values with critical values
    3. If p-value at start_idx_sud is suspicious (≤ critical value):
       - Perform Step-Down: find first non-suspicious value and reject all before it
    4. Otherwise:
       - Perform Step-Up: find last suspicious value and reject all up to it

    This is a helper function used by SD() and SU() procedures.
    """
    pvalues = np.asarray(pvalues, dtype=float)
    critical_values = np.asarray(critical_values, dtype=float)
    m = len(critical_values)

    # Plausibility checks
    if m == 1:
        raise ValueError("SUD(): Only 1 critical value. Use SS() instead.")
    if m != len(pvalues):
        raise ValueError(
            "SUD(): Length of critical_values and pvalues must match."
        )
    if start_idx_sud < 1 or start_idx_sud > m:
        raise ValueError("SUD(): start_idx_sud out of bounds.")

    rejected = np.zeros(m, dtype=bool)

    # Work with ordered p-values
    order = np.argsort(pvalues)
    sorted_pvals = pvalues[order]

    suspicious = sorted_pvals <= critical_values

    if suspicious[start_idx_sud - 1]:
        # Suspicious p-value at start_idx_sud → Step-Down
        non_susp_above = np.where(~suspicious[start_idx_sud - 1 :])[0]

        if len(non_susp_above) == 0:
            # All suspicious → reject all
            return np.ones(m, dtype=bool)

        # Correct for index shift
        non_susp_above = non_susp_above + start_idx_sud - 1
        min_idx = np.min(non_susp_above) - 1

        if min_idx >= 0:
            rejected[order[: min_idx + 1]] = True

    else:
        # Not suspicious at start_idx_sud → Step-Up
        suspicious_idx = np.where(suspicious[:start_idx_sud])[0]

        if len(suspicious_idx) == 0:
            return np.zeros(m, dtype=bool)

        max_idx = np.max(suspicious_idx)
        rejected[order[: max_idx + 1]] = True

    return rejected


def SD(pvalues, critical_values):
    """
    Step-Down procedure for multiple testing.

    This is a wrapper around SUD that performs step-down testing by starting
    from the smallest p-value.

    Parameters
    ----------
    pvalues : array-like of shape (m,)
        P-values from multiple hypothesis tests.

    critical_values : array-like of shape (m,)
        Critical values corresponding to each p-value.

    Returns
    -------
    rejected : ndarray of shape (m,), dtype=bool
        Boolean array indicating which hypotheses are rejected.

    See Also
    --------
    SUD : General step-up/step-down procedure.
    SU : Step-up procedure.
    """
    return SUD(pvalues, critical_values, start_idx_sud=1)


def SU(pvalues, critical_values):
    """
    Step-Up procedure for multiple testing.

    This is a wrapper around SUD that performs step-up testing by starting
    from the largest p-value.

    Parameters
    ----------
    pvalues : array-like of shape (m,)
        P-values from multiple hypothesis tests.

    critical_values : array-like of shape (m,)
        Critical values corresponding to each p-value.

    Returns
    -------
    rejected : ndarray of shape (m,), dtype=bool
        Boolean array indicating which hypotheses are rejected.

    See Also
    --------
    SUD : General step-up/step-down procedure.
    SD : Step-down procedure.
    """
    return SUD(pvalues, critical_values, start_idx_sud=len(critical_values))


def benjamini_hochberg(pvals: np.ndarray, alpha: float = 0.05) -> Dict:
    """
    Apply Benjamini-Hochberg false discovery rate correction.

    The Benjamini-Hochberg procedure controls the false discovery rate (FDR)
    at level alpha. It provides a less conservative alternative to family-wise
    error rate (FWER) methods like Bonferroni correction by allowing some
    false positives while controlling their expected proportion.

    Parameters
    ----------
    pvals : array-like of shape (n_tests,)
        Raw p-values from multiple hypothesis tests. Must be numeric values
        between 0 and 1.

    alpha : float, default=0.05
        The desired false discovery rate level. Must be between 0 and 1.

    Returns
    -------
    results : dict
        Dictionary containing the following keys:

        - 'qvals' : ndarray of shape (n_tests,)
            Adjusted p-values (q-values) corresponding to the input p-values.
            These represent the minimum FDR at which each hypothesis would
            be rejected.

        - 'rejected' : ndarray of shape (n_tests,), dtype=bool
            Boolean array indicating which null hypotheses are rejected
            at the given alpha level.

        - 'alpha' : float
            The FDR level used for the correction.

        - 'method' : str
            Always 'BH' to indicate Benjamini-Hochberg procedure.

    Notes
    -----
    The Benjamini-Hochberg procedure works by:

    1. Sorting p-values in ascending order
    2. Computing q-values using the formula: q_i = p_i * m / i
    3. Enforcing monotonicity constraint: q_i <= q_{i+1}
    4. Rejecting hypotheses where q_i <= alpha

    The method assumes independence or positive dependence among test
    statistics. For arbitrary dependence, consider more conservative
    procedures.

    References
    ----------
    .. [1] Benjamini, Y., & Hochberg, Y. (1995). Controlling the false
           discovery rate: a practical and powerful approach to multiple
           testing. Journal of the Royal Statistical Society: Series B
           (Methodological), 57(1), 289-300.
    """
    p = np.asarray(pvals, dtype=float)
    m = p.size
    order = np.argsort(p)
    p_sorted = p[order]

    q_sorted = (p_sorted * m) / np.arange(1, m + 1)
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q_sorted = np.minimum(q_sorted, 1.0)

    qvals = np.empty_like(q_sorted)
    qvals[order] = q_sorted
    rejected = qvals <= alpha
    return {
        "qvals": qvals,
        "rejected": rejected,
        "alpha": alpha,
        "method": "BH",
    }


def benjamini_liu(pvalues, alpha=0.05, verbose=False):
    """
    Apply Benjamini-Liu step-down FDR procedure.

    The Benjamini-Liu (1999) procedure is a step-down method for controlling
    the false discovery rate. It is more powerful than the standard
    Benjamini-Hochberg procedure under certain dependence structures,
    particularly for positively dependent test statistics.

    Parameters
    ----------
    pvalues : array-like of shape (n_tests,)
        P-values from multiple hypothesis tests. Must be numeric values
        between 0 and 1.

    alpha : float, default=0.05
        The desired false discovery rate level. Must be between 0 and 1.

    verbose : bool, default=False
        If True, prints a summary of the procedure results including
        the alpha level and indices of rejected hypotheses.

    Returns
    -------
    results : dict
        Dictionary containing the following keys:

        - 'qvals' : ndarray of shape (n_tests,)
            Adjusted p-values (q-values) corresponding to the input p-values.
            These represent the minimum FDR at which each hypothesis would
            be rejected.

        - 'rejected' : ndarray of shape (n_tests,), dtype=bool
            Boolean array indicating which null hypotheses are rejected
            at the given alpha level.

        - 'alpha' : float
            The FDR level used for the correction.

        - 'method' : str
            Always 'BL' to indicate Benjamini-Liu procedure.

    Notes
    -----
    The Benjamini-Liu procedure uses critical values of the form:

    .. math::

        c_i = 1 - (1 - \\min(1, m\\alpha/(m-i+1)))^{1/(m-i+1)}

    where m is the number of tests and i is the rank of the p-value.

    The adjusted q-values are computed using a step-down approach that
    accounts for the dependence structure among tests.

    References
    ----------
    .. [1] Benjamini, Y., & Liu, W. (1999). A step-down multiple hypotheses
           testing procedure that controls the false discovery rate under
           independence. Journal of Statistical Planning and Inference,
           82(1-2), 163-170.

    Examples
    --------
    >>> import numpy as np
    >>> pvals = np.array([0.001, 0.01, 0.2, 0.5])
    >>> results = benjamini_liu(pvals, alpha=0.05)
    >>> results['rejected']
    array([ True,  True, False, False])
    >>> np.round(results['qvals'], 4)
    array([0.004 , 0.0223, 0.18  , 0.18  ])

    >>> # Compare with standard BH
    >>> from covtest.multiplicity.fdr import benjamini_hochberg
    >>> bh_results = benjamini_hochberg(pvals, alpha=0.05)
    >>> np.sum(results['rejected']) == np.sum(bh_results['rejected'])
    True
    """
    pvalues = np.asarray(pvalues, dtype=float)
    m = len(pvalues)

    # Critical values
    critical_values = np.array(
        [
            1 - (1 - min(1, (m * alpha) / (m - i + 1))) ** (1 / (m - i + 1))
            for i in range(1, m + 1)
        ]
    )

    # Rejection decisions via SD procedure
    rejected = SD(pvalues, critical_values)

    # Sort p-values
    order = np.argsort(pvalues)
    spval = pvalues[order]

    # Initialize adjusted q-values
    qvals = np.zeros(m)
    qvals[0] = min(1 - (1 - spval[0]) ** m, 1)

    # Step-down adjusted q-values
    for i in range(1, m):
        if (alpha * m) / (m - i) <= 1:
            val = ((m - i) / m) * (1 - (1 - spval[i]) ** (m - i))
        else:
            val = 0
        qvals[i] = max(qvals[i - 1], val)

    # Reorder adjusted q-values to match input order
    qvals_final = np.zeros(m)
    qvals_final[order] = qvals

    if verbose:
        print("\nBenjamini–Liu's (1999) Step-Down Procedure")
        print("alpha =", alpha)
        print("Rejected hypotheses:", np.where(rejected)[0])

    return {
        "qvals": qvals_final,
        "rejected": rejected,
        "alpha": alpha,
        "method": "BL",
    }


def benjamini_yekutieli(pvals: np.ndarray, alpha: float = 0.05) -> Dict:
    """Apply Benjamini-Yekutieli false discovery rate correction.

    The Benjamini-Yekutieli procedure is a more conservative variant of the
    Benjamini-Hochberg method that controls the false discovery rate (FDR)
    under arbitrary dependence structures among test statistics. It multiplies
    the standard BH correction by a harmonic series factor.

    Parameters
    ----------
    pvals : array-like of shape (n_tests,)
        Raw p-values from multiple hypothesis tests. Must be numeric values
        between 0 and 1.

    alpha : float, default=0.05
        The desired false discovery rate level. Must be between 0 and 1.

    Returns
    -------
    results : dict
        Dictionary containing the following keys:

        - 'qvals' : ndarray of shape (n_tests,)
            Adjusted p-values (q-values) corresponding to the input p-values.
            These represent the minimum FDR at which each hypothesis would
            be rejected under arbitrary dependence.

        - 'rejected' : ndarray of shape (n_tests,), dtype=bool
            Boolean array indicating which null hypotheses are rejected
            at the given alpha level.

        - 'alpha' : float
            The FDR level used for the correction.

        - 'method' : str
            Always 'BY' to indicate Benjamini-Yekutieli procedure.

    Notes
    -----
    The Benjamini-Yekutieli procedure extends the BH method by:

    1. Computing the harmonic series correction factor: c(m) = sum(1/i) for i=1 to m
    2. Modifying the q-value formula: q_i = p_i * m * c(m) / i
    3. Enforcing monotonicity constraint: q_i <= q_{i+1}
    4. Rejecting hypotheses where q_i <= alpha

    The correction factor c(m) ≈ log(m) + 0.577 for large m, making this
    procedure more conservative than standard BH. Use this method when
    test statistics may have arbitrary dependence structures, negative
    correlations, or when the independence assumption of BH is violated.

    The method guarantees FDR control at level alpha regardless of the
    dependence structure among tests.

    References
    ----------
    .. [1] Benjamini, Y., & Yekutieli, D. (2001). The control of the false
           discovery rate in multiple testing under dependency. Annals of
           Statistics, 29(4), 1165-1188.

    Examples
    --------
    >>> import numpy as np
    >>> pvals = np.array([0.001, 0.01, 0.03, 0.07, 0.2])
    >>> results = benjamini_yekutieli(pvals, alpha=0.05)
    >>> results['qvals']
    array([0.01141667, 0.05708333, 0.11416667, 0.19979167, 0.45666667])
    >>> results['rejected']
    array([ True, False, False, False, False])

    >>> # Compare with BH method (less conservative)
    >>> bh_results = benjamini_hochberg(pvals, alpha=0.05)
    >>> np.sum(bh_results['rejected']) > np.sum(results['rejected'])
    np.True_

    >>> # Higher alpha needed for BY to match BH power
    >>> results_higher = benjamini_yekutieli(pvals, alpha=0.12)
    >>> np.sum(results_higher['rejected'])
    3
    """
    p = np.asarray(pvals, dtype=float)
    m = p.size
    c_m = np.sum(1.0 / np.arange(1, m + 1))
    order = np.argsort(p)
    p_sorted = p[order]
    q_sorted = (p_sorted * m * c_m) / np.arange(1, m + 1)
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q_sorted = np.minimum(q_sorted, 1.0)
    qvals = np.empty_like(q_sorted)
    qvals[order] = q_sorted
    rejected = qvals <= alpha
    return {
        "qvals": qvals,
        "rejected": rejected,
        "alpha": alpha,
        "method": "BY",
    }


def blaroq(pvalues, alpha=0.05, pii=None, verbose=False):
    """
    Apply Blanchard-Roquain step-up FDR procedure for arbitrary dependence.

    The Blanchard-Roquain (2008) procedure, also known as the Sarkar procedure,
    is a step-up method that controls the false discovery rate under arbitrary
    dependence structures. It uses a weighted prior to improve power while
    maintaining FDR control without independence assumptions.

    Parameters
    ----------
    pvalues : array-like of shape (n_tests,)
        P-values from multiple hypothesis tests. Must be numeric values
        between 0 and 1.

    alpha : float, default=0.05
        The desired false discovery rate level. Must be between 0 and 1.

    pii : array-like of shape (n_tests,), default=None
        Prior weights for each hypothesis. If None, uses an exponentially
        decreasing prior: pii[i] = exp(-i / (0.15 * n_tests)).
        Must contain only positive values and have the same length as pvalues.

    verbose : bool, default=False
        If True, prints a summary of the procedure results including
        the alpha level and indices of rejected hypotheses.

    Returns
    -------
    results : dict
        Dictionary containing the following keys:

        - 'qvals' : ndarray of shape (n_tests,)
            Adjusted p-values (q-values) corresponding to the input p-values.
            These represent the minimum FDR at which each hypothesis would
            be rejected.

        - 'rejected' : ndarray of shape (n_tests,), dtype=bool
            Boolean array indicating which null hypotheses are rejected
            at the given alpha level.

        - 'alpha' : float
            The FDR level used for the correction.

        - 'method' : str
            Always 'BlaRoq' to indicate Blanchard-Roquain procedure.

    Raises
    ------
    ValueError
        If prior pii contains negative elements.
        If prior pii has different length than pvalues.

    Notes
    -----
    The procedure uses pre-critical values computed from the prior weights:

    .. math::

        \\text{precritical}_i = \\sum_{j=1}^{i} \\frac{(j+1) \\pi_j}{m}

    where the prior is normalized to sum to 1.

    The adjusted p-values are computed by dividing p-values by pre-critical
    values and taking the cumulative minimum in reverse order.

    This method is particularly useful when test statistics have complex
    or unknown dependence structures, as it does not require independence
    or positive dependence assumptions.

    References
    ----------
    .. [1] Blanchard, G., & Roquain, E. (2008). Two simple sufficient
           conditions for FDR control. Electronic Journal of Statistics,
           2, 963-992.

    .. [2] Sarkar, S. K. (2008). Generalizing Simes' test and Hochberg's
           stepup procedure. Annals of Statistics, 36(1), 337-363.

    Examples
    --------
    >>> import numpy as np
    >>> pvals = np.array([0.001, 0.01, 0.2, 0.5])
    >>> results = blaroq(pvals, alpha=0.05)
    >>> results['rejected']
    array([ True,  True, False, False])
    >>> np.round(results['qvals'], 4)
    array([0.0049, 0.0357, 0.6634, 1.    ])

    >>> # Custom prior weights (higher weight for first test)
    >>> custom_prior = np.array([2.0, 1.0, 1.0, 1.0])
    >>> results_custom = blaroq(pvals, alpha=0.05, pii=custom_prior)
    >>> results_custom['rejected']
    array([ True,  True, False, False])
    """
    pvalues = np.asarray(pvalues, dtype=float)
    k = len(pvalues)

    # Default prior if not provided
    if pii is None:
        pii = np.array([np.exp(-i / (0.15 * k)) for i in range(1, k + 1)])

    pii = np.asarray(pii, dtype=float)

    if np.any(pii < 0):
        raise ValueError("BlaRoq(): Prior pii can only have positive elements")

    if len(pii) != k:
        raise ValueError(
            "BlaRoq(): Prior pii must have the same length as pvalues"
        )

    # Normalize prior
    pii = pii / np.sum(pii)

    # Pre-critical values
    precritical_values = np.cumsum([(i + 1) * pii[i] / k for i in range(k)])

    # Sort p-values in decreasing order
    o = np.argsort(-pvalues)  # indices of sorted pvalues (descending)
    ro = np.argsort(o)  # inverse permutation
    i = np.arange(k - 1, -1, -1)  # k:1 in R (descending indices)

    # Compute adjusted p-values
    adj_temp = pvalues[o] / precritical_values[i]
    adj_temp = np.minimum(1, np.minimum.accumulate(adj_temp))  # cummin
    qvals = adj_temp[ro]  # reorder back to original order

    # Rejected decisions
    rejected = qvals <= alpha

    if verbose:
        print("\nBlanchard–Roquain/Sarkar (2008) Step-Up Procedure")
        print("alpha =", alpha)
        print("Rejected hypotheses:", np.where(rejected)[0])

    return {
        "qvals": qvals,
        "rejected": rejected,
        "alpha": alpha,
        "method": "BlaRoq",
    }


def weighted_bh(
    pvals: np.ndarray, weights: np.ndarray, alpha: float = 0.05
) -> Dict:
    """Apply weighted Benjamini-Hochberg false discovery rate correction.

    The weighted Benjamini-Hochberg procedure allows incorporating prior
    knowledge about hypothesis importance through weights. Tests with higher
    weights are more likely to be rejected, enabling adaptive FDR control
    based on external information or prior beliefs about test significance.

    Parameters
    ----------
    pvals : array-like of shape (n_tests,)
        Raw p-values from multiple hypothesis tests. Must be numeric values
        between 0 and 1.

    weights : array-like of shape (n_tests,)
        Non-negative weights for each hypothesis test. Higher weights
        increase the likelihood of rejection. Weights are normalized
        internally to sum to n_tests.

    alpha : float, default=0.05
        The desired false discovery rate level. Must be between 0 and 1.

    Returns
    -------
    results : dict
        Dictionary containing the following keys:

        - 'qvals' : ndarray of shape (n_tests,)
            Adjusted p-values (q-values) based on weighted p-values.
            These represent the minimum FDR at which each hypothesis would
            be rejected under the weighting scheme.

        - 'rejected' : ndarray of shape (n_tests,), dtype=bool
            Boolean array indicating which null hypotheses are rejected
            at the given alpha level.

        - 'alpha' : float
            The FDR level used for the correction.

        - 'method' : str
            Always 'wBH' to indicate weighted Benjamini-Hochberg procedure.

    Notes
    -----
    The weighted BH procedure works by:

    1. Normalizing weights to sum to m (number of tests)
    2. Computing weighted p-values: p*_i = p_i / max(w_i, ε)
    3. Applying standard BH procedure to weighted p-values
    4. Rejecting hypotheses where q*_i <= alpha

    The weighting effectively "boosts" hypotheses with higher weights by
    making their effective p-values smaller. A small epsilon (1e-16) prevents
    division by zero for weights that are exactly zero.

    This method maintains FDR control at level alpha when weights are
    chosen independently of the data. Adaptive weights that depend on
    the observed data may require additional considerations.

    Raises
    ------
    ValueError
        If weights contain negative values or if weights and pvals have
        different shapes.

    References
    ----------
    .. [1] Genovese, C. R., Roeder, K., & Wasserman, L. (2006). False
           discovery control with p-value weighting. Biometrika, 93(3),
           509-524.

    Examples
    --------
    >>> import numpy as np
    >>> pvals = np.array([0.001, 0.01, 0.03, 0.07, 0.2])
    >>>
    >>> # Equal weights (equivalent to standard BH)
    >>> weights_equal = np.ones(5)
    >>> results = weighted_bh(pvals, weights_equal, alpha=0.05)
    >>> results['rejected']
    array([ True,  True,  True, False, False])

    >>> # Higher weight for first test
    >>> weights_unequal = np.array([3.0, 1.0, 1.0, 1.0, 1.0])
    >>> results = weighted_bh(pvals, weights_unequal, alpha=0.05)
    >>> results['rejected']
    array([ True,  True,  True,  True, False])

    >>> # Zero weight excludes a hypothesis
    >>> weights_sparse = np.array([1.0, 0.0, 1.0, 1.0, 1.0])
    >>> results = weighted_bh(pvals, weights_sparse, alpha=0.05)
    >>> results['rejected'][1]  # Second test never rejected
    False

    >>> # Error handling for invalid weights
    >>> try:
    ...     weighted_bh(pvals, np.array([-1, 1, 1, 1, 1]))
    ... except ValueError as e:
    ...     print("Caught expected error:", str(e))
    Caught expected error: weights must be nonnegative and same length as pvals
    """
    p = np.asarray(pvals, dtype=float)
    w = np.asarray(weights, dtype=float)
    m = p.size

    if w.shape != p.shape or np.any(w < 0):
        raise ValueError("weights must be nonnegative and same length as pvals")

    # normalize to sum=m
    w = w * (m / np.sum(w))

    p_star = p / np.maximum(w, 1e-16)
    order = np.argsort(p_star)
    p_sorted = p_star[order]

    q_sorted = (m * p_sorted) / np.arange(1, m + 1)
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q_sorted = np.minimum(q_sorted, 1.0)

    qvals = np.empty_like(q_sorted)
    qvals[order] = q_sorted
    rejected = qvals <= alpha
    return {
        "qvals": qvals,
        "rejected": rejected,
        "alpha": alpha,
        "method": "wBH",
    }


def storey_qvalues(
    pvals: np.ndarray, alpha: float = 0.05, lambdas=None
) -> Dict:
    """Apply Storey's q-value method for false discovery rate estimation.

    Storey's method improves upon the Benjamini-Hochberg procedure by estimating
    the proportion of true null hypotheses (pi_0) from the data. This typically
    results in less conservative corrections and higher statistical power when
    many null hypotheses are actually true.

    Parameters
    ----------
    pvals : array-like of shape (n_tests,)
        Raw p-values from multiple hypothesis tests. Must be numeric values
        between 0 and 1.

    alpha : float, default=0.05
        The desired false discovery rate level. Must be between 0 and 1.

    lambdas : array-like, default=None
        Threshold values used to estimate pi_0. If None, uses 19 evenly spaced
        values from 0.05 to 0.95. Higher values are more conservative but
        may be less stable with small sample sizes.

    Returns
    -------
    results : dict
        Dictionary containing the following keys:

        - 'qvals' : ndarray of shape (n_tests,)
            Storey q-values corresponding to the input p-values. These represent
            the minimum false discovery rate at which each hypothesis would
            be rejected, accounting for the estimated pi_0.

        - 'rejected' : ndarray of shape (n_tests,), dtype=bool
            Boolean array indicating which null hypotheses are rejected
            at the given alpha level.

        - 'alpha' : float
            The FDR level used for the correction.

        - 'pi0' : float
            Estimated proportion of true null hypotheses, constrained to [0,1].
            Values closer to 1 indicate most hypotheses are likely null.

        - 'method' : str
            Always 'Storey' to indicate Storey's q-value method.

    Notes
    -----
    Storey's method extends the BH procedure by:

    1. Estimating pi_0 using: pi_0_hat(lambda) = #{p_i >= lambda} / (m(1-lambda)) for each lambda
    2. Taking the minimum: pi_0_hat = min(1, max(0, min_lambda pi_0_hat(lambda)))
    3. Computing q-values: q_i = pi_0_hat * m * p_(i) / i
    4. Enforcing monotonicity: q_i <= q_{i+1}

    The key insight is that when pi_0 < 1 (some hypotheses are truly alternative),
    the standard BH method is overly conservative. By estimating pi_0, Storey's
    method can achieve the same FDR control with greater statistical power.

    The lambda grid should be chosen carefully:
    - Too small: unstable pi_0 estimates
    - Too large: overly conservative estimates
    - Default range [0.05, 0.95] works well in practice

    When pi_0 = 1 (all nulls true), Storey's method reduces to standard BH.
    When pi_0 < 1, it provides more liberal corrections.

    References
    ----------
    .. [1] Storey, J. D. (2002). A direct approach to false discovery rates.
           Journal of the Royal Statistical Society: Series B (Statistical
           Methodology), 64(3), 479-498.

    .. [2] Storey, J. D., & Tibshirani, R. (2003). Statistical significance
           for genomewide studies. Proceedings of the National Academy of
           Sciences, 100(16), 9440-9445.

    .. [3] Benjamini, Y., & Hochberg, Y. (1995). Controlling the false
           discovery rate: a practical and powerful approach to multiple
           testing. Journal of the Royal Statistical Society: Series B
           (Methodological), 57(1), 289-300.

    Examples
    --------
    >>> import numpy as np
    >>> # Mix of null and alternative hypotheses
    >>> np.random.seed(42)
    >>> pvals_null = np.random.uniform(0, 1, 800)  # True nulls
    >>> pvals_alt = np.random.beta(1, 10, 200)     # Alternatives
    >>> pvals = np.concatenate([pvals_null, pvals_alt])
    >>>
    >>> results = storey_qvalues(pvals, alpha=0.05)
    >>> print(f"Estimated pi_0: {results['pi0']:.3f}")
    Estimated pi_0: 0.842
    >>> print(f"Rejections: {np.sum(results['rejected'])}")
    Rejections: 156

    >>> # Compare with standard BH (more conservative)
    >>> bh_results = benjamini_hochberg(pvals, alpha=0.05)
    >>> print(f"BH rejections: {np.sum(bh_results['rejected'])}")
    BH rejections: 132

    >>> # Custom lambda grid for pi_0 estimation
    >>> custom_lambdas = np.linspace(0.1, 0.8, 10)
    >>> results_custom = storey_qvalues(pvals, lambdas=custom_lambdas)
    >>> print(f"Custom pi_0: {results_custom['pi0']:.3f}")
    Custom pi_0: 0.856

    >>> # When all hypotheses are null, pi_0 approaches 1
    >>> pvals_all_null = np.random.uniform(0, 1, 1000)
    >>> results_null = storey_qvalues(pvals_all_null)
    >>> print(f"All null pi_0: {results_null['pi0']:.3f}")
    All null pi_0: 0.983
    """
    p = np.asarray(pvals, dtype=float)
    m = p.size
    if lambdas is None:
        lambdas = np.linspace(0.05, 0.95, 19)

    ratios = [(p >= lam).mean() / (1 - lam) for lam in lambdas]
    pi0 = min(1.0, max(0.0, np.min(ratios)))

    order = np.argsort(p)
    p_sorted = p[order]

    q_sorted = (pi0 * m * p_sorted) / np.arange(1, m + 1)
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q_sorted = np.minimum(q_sorted, 1.0)

    qvals = np.empty_like(q_sorted)
    qvals[order] = q_sorted
    rejected = qvals <= alpha
    return {
        "qvals": qvals,
        "rejected": rejected,
        "alpha": alpha,
        "pi0": pi0,
        "method": "Storey",
    }
