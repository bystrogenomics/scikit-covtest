import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression


def analyze_pvalues(p_values, num_permutations=1000, seed=42):
    """
    Perform comprehensive diagnostic analysis of p-value distributions.

    This function applies multiple statistical tests and diagnostics to assess
    whether a set of p-values follows a uniform distribution under the null
    hypothesis. It includes formal tests, deviation metrics, inflation factors,
    tail enrichment tests, and QQ-plot regression.

    Parameters
    ----------
    p_values : array-like of shape (n_tests,)
        Array of p-values to analyze. Values should be between 0 and 1.

    num_permutations : int, default=1000
        Number of permutation replicates for computing empirical p-values
        of the ECDF deviation statistic.

    seed : int, default=42
        Random seed for reproducibility of permutation tests.

    Returns
    -------
    results : dict
        Dictionary containing the following diagnostic results:

        - 'ks' : dict
            Kolmogorov-Smirnov test results with keys:

            - 'stat' : float
                KS test statistic.
            - 'pval' : float
                P-value from KS test.

        - 'ad' : dict
            Anderson-Darling test results with keys:

            - 'stat' : float
                AD test statistic for uniform distribution.

        - 'ecdf_deviation' : dict
            ECDF deviation metrics with keys:

            - 'linf' : float
                L-infinity (maximum absolute) deviation.
            - 'l2' : float
                L2 (root mean squared) deviation.

        - 'inflation_factor' : float
            Genomic inflation factor (λGC) computed from median chi-squared
            statistic. Values > 1 indicate inflation.

        - 'storey_pi0' : float
            Storey's estimate of the proportion of true null hypotheses.
            Values < 1 suggest presence of true signals.

        - 'tail_tests' : dict
            Tail enrichment tests for thresholds [0.05, 0.01, 0.001].
            Each threshold maps to a dict with:

            - 'count' : int
                Observed number of p-values below threshold.
            - 'expected' : float
                Expected count under uniform distribution.
            - 'binom_p' : float
                Binomial test p-value for enrichment.

        - 'qq_fit' : dict
            QQ-plot linear regression results with keys:

            - 'intercept' : float
                Intercept of -log10(p) QQ-plot regression.
            - 'slope' : float
                Slope of -log10(p) QQ-plot regression.

        - 'perm_l2_pval' : float
            Permutation-based p-value for L2 ECDF deviation.

    Notes
    -----
    **Kolmogorov-Smirnov Test**
        Tests whether p-values follow a uniform distribution using the
        maximum deviation between empirical and theoretical CDFs.

    **Anderson-Darling Test**
        More sensitive than KS test to deviations in the tails. Computed
        using a uniform-specific formula.

    **Inflation Factor (λGC)**
        Ratio of median observed chi-squared statistic to expected median
        under the null. Values > 1 indicate systematic inflation of test
        statistics.

    **Storey's π₀**
        Estimates the proportion of true null hypotheses using p-values
        above a threshold (default 0.5). Lower values suggest more true
        signals.

    **Tail Enrichment**
        Tests whether small p-values are more frequent than expected under
        the null using binomial tests at multiple thresholds.

    **QQ-plot Regression**
        Fits a line to the QQ-plot of -log10(p-values). Intercept near 0
        and slope near 1 indicate good calibration.

    References
    ----------
    .. [1] Storey, J. D., & Tibshirani, R. (2003). Statistical significance
           for genomewide studies. Proceedings of the National Academy of
           Sciences, 100(16), 9440-9445.

    .. [2] Devlin, B., & Roeder, K. (1999). Genomic control for association
           studies. Biometrics, 55(4), 997-1004.

    .. [3] Anderson, T. W., & Darling, D. A. (1952). Asymptotic theory of
           certain "goodness of fit" criteria based on stochastic processes.
           The Annals of Statistical Mathematics, 23(2), 193-212.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> # Well-calibrated p-values (uniform)
    >>> p_uniform = np.random.uniform(0, 1, 1000)
    >>> results = analyze_pvalues(p_uniform)
    >>> results['ks']['pval'] > 0.05  # Should not reject uniformity
    True
    >>> abs(results['inflation_factor'] - 1.0) < 0.1  # Should be near 1
    True

    >>> # Inflated p-values (too many small values)
    >>> p_inflated = np.random.beta(0.5, 2, 1000)
    >>> results_inf = analyze_pvalues(p_inflated)
    >>> results_inf['inflation_factor'] > 1.0  # Indicates inflation
    True
    >>> results_inf['storey_pi0'] < 0.9  # Suggests true signals
    True
    """
    rng = np.random.default_rng(seed)
    n = len(p_values)
    p_values = np.asarray(p_values)

    # --- Formal Tests ---
    ks_stat, ks_pval = stats.kstest(p_values, "uniform")

    # Anderson-Darling-like statistic for Uniform[0,1]
    def anderson_darling_uniform(pvals):
        """Compute Anderson-Darling statistic for uniform distribution.

        Parameters
        ----------
        pvals : ndarray
            P-values to test.

        Returns
        -------
        float
            Anderson-Darling test statistic.
        """
        sorted_p = np.sort(pvals)
        i = np.arange(1, len(pvals) + 1)
        term1 = (2 * i - 1) * np.log(sorted_p)
        term2 = (2 * (len(pvals) - i) + 1) * np.log(1 - sorted_p[::-1])
        return -len(pvals) - np.mean(term1 + term2)

    ad_stat = anderson_darling_uniform(p_values)

    # --- Scalar Deviation ---
    sorted_p = np.sort(p_values)
    ecdf = np.arange(1, n + 1) / n
    linf_dev = np.max(np.abs(ecdf - sorted_p))
    l2_dev = np.sqrt(np.mean((ecdf - sorted_p) ** 2))

    # --- Inflation Factor (λGC) ---
    chisq = stats.chi2.isf(p_values, df=1)
    lambda_gc = np.median(chisq) / stats.chi2.ppf(0.5, df=1)

    # --- Storey’s π₀ Estimator ---
    lambda_val = 0.5
    pi0 = np.sum(p_values > lambda_val) / ((1 - lambda_val) * n)
    pi0 = min(pi0, 1.0)

    # --- Tail Enrichment ---
    tail_thresholds = [0.05, 0.01, 0.001]
    tail_tests = {}
    for alpha in tail_thresholds:
        count = np.sum(p_values < alpha)
        expected = alpha * n
        pval = stats.binomtest(count, n, alpha, alternative="greater").pvalue
        tail_tests[alpha] = {
            "count": count,
            "expected": expected,
            "binom_p": pval,
        }

    # --- QQ Intercept and Slope ---
    expected = -np.log10(np.linspace(1 / (n + 1), 1, n))
    observed = -np.log10(np.clip(np.sort(p_values), np.finfo(float).tiny, 1.0))
    reg = LinearRegression().fit(expected.reshape(-1, 1), observed)
    intercept = reg.intercept_
    slope = reg.coef_[0]

    # --- Permutation-based ECDF deviation ---
    perm_stats = []
    for _ in range(num_permutations):
        perm = rng.uniform(0, 1, n)
        perm_sorted = np.sort(perm)
        perm_ecdf = np.arange(1, n + 1) / n
        stat = np.sqrt(np.mean((perm_ecdf - perm_sorted) ** 2))
        perm_stats.append(stat)
    perm_stats = np.array(perm_stats)
    perm_pval = np.mean(perm_stats >= l2_dev)

    return {
        "ks": {"stat": ks_stat, "pval": ks_pval},
        "ad": {"stat": ad_stat},
        "ecdf_deviation": {"linf": linf_dev, "l2": l2_dev},
        "inflation_factor": lambda_gc,
        "storey_pi0": pi0,
        "tail_tests": tail_tests,
        "qq_fit": {"intercept": intercept, "slope": slope},
        "perm_l2_pval": perm_pval,
    }
