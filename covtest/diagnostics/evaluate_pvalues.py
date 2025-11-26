import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression


def analyze_pvalues(p_values, num_permutations=1000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(p_values)
    p_values = np.asarray(p_values)

    # --- Formal Tests ---
    ks_stat, ks_pval = stats.kstest(p_values, "uniform")

    # Anderson-Darling-like statistic for Uniform[0,1]
    def anderson_darling_uniform(pvals):
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
    observed = -np.log10(np.sort(p_values))
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
