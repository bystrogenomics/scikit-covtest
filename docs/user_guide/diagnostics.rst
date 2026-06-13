.. _diagnostics:

Assumption Verification and Diagnostics
=======================================

Before selecting and running covariance matrix tests, it is critical to verify the statistical assumptions of the data and evaluate the performance of your test statistics. ``scikit-covtest`` provides a comprehensive set of diagnostic tools split into two modules:

1. **Assumptions and Normality Diagnostics**: Assessing multivariate normality, rank, conditioning, and eigenvalue spectra.
2. **P-Value Calibration Diagnostics**: Evaluating the distribution of p-values to verify false positive control and detect inflation/deflation.

All diagnostics are implemented in the :mod:`covtest.diagnostics` submodule.

Multivariate Normality Tests
----------------------------

Many classical covariance tests assume the underlying data is multivariate normal. ``scikit-covtest`` implements three widely-used tests for multivariate normality in :mod:`covtest.diagnostics.assumptions`:

- **Mardia's Tests** (``mardia_tests``): Evaluates multivariate skewness and kurtosis. Can use asymptotic distributions or parametric bootstrap calibration.
- **Henze-Zirkler Test** (``hz_test``): An omnibus test based on empirical characteristic functions. High power against a broad range of non-normal alternatives.
- **Royston's Test** (``royston_test``): Combines Shapiro-Francia univariate normality tests with a correlation adjustment.

**Example Usage**:

.. code-block:: python

   import numpy as np
   from covtest.diagnostics.assumptions import mardia_tests, hz_test

   rng = np.random.default_rng(42)
   # Generate multivariate normal data (n=100, p=3)
   data_normal = rng.multivariate_normal([0, 0, 0], np.eye(3), size=100)

   # Mardia's skewness and kurtosis
   mardia_res = mardia_tests(data_normal)
   print("Mardia skewness p-value:", mardia_res['p_value'][0])
   print("Mardia kurtosis p-value:", mardia_res['p_value'][1])

   # Henze-Zirkler test
   hz_res = hz_test(data_normal)
   print("Henze-Zirkler p-value:", hz_res['p_value'])

Eigenvalue Spectrum & Conditioning
----------------------------------

High-dimensional covariance estimation and testing are heavily influenced by the eigenvalue structure and numerical properties of the sample covariance matrix.

- **Eigenvalue Spectrum** (``eigen_spectrum``): Computes and plots the sorted eigenvalues of the sample covariance matrix. It can optionally overlay the Marchenko–Pastur (MP) bulk limits, which represent the theoretical distribution of eigenvalues under the null hypothesis of identity covariance (white noise).
- **Condition and Rank Analysis** (``condition_and_rank``): Computes the spectral condition number (ratio of largest to smallest eigenvalue), numerical rank (eigenvalues above a threshold), and entropy-based effective rank.

**Example Usage**:

.. code-block:: python

   import numpy as np
   from covtest.diagnostics.assumptions import eigen_spectrum, condition_and_rank

   rng = np.random.default_rng(42)
   X = rng.normal(size=(200, 50))  # n=200 samples, p=50 features

   # Compute condition number and effective rank
   info = condition_and_rank(X)
   print("Condition Number:", info['condition_number'])
   print("Effective Rank:", info['effective_rank'])

   # Get eigenvalues and MP limits (without plotting/displaying)
   spec = eigen_spectrum(X, plot=False)
   print("Largest eigenvalue:", spec['eigenvalues'][0])
   print("Marchenko-Pastur bulk max:", spec['mp_max'])

Analyzing P-Values
------------------

When executing large-scale simulation studies or family-wise hypothesis tests, you expect null p-values to follow a Uniform(0, 1) distribution. The ``analyze_pvalues`` function in :mod:`covtest.diagnostics.evaluate_pvalues` performs a comprehensive suite of tests on a collection of p-values:

- **Goodness-of-Fit Tests**: Kolmogorov-Smirnov (KS) and Anderson-Darling (AD) tests check for deviations from a uniform distribution.
- **Genomic Inflation Factor ($\lambda_{GC}$)**: Computes the ratio of the median observed chi-squared statistic to the expected median. $\lambda_{GC} > 1$ indicates inflation (inflated false positive rates), while $\lambda_{GC} < 1$ indicates deflation (conservative tests).
- **Storey's $\pi_0$**: Estimates the proportion of true null hypotheses in the set.
- **Tail Enrichment Tests**: Performs binomial tests at thresholds of 0.05, 0.01, and 0.001 to detect excess small p-values.
- **QQ-Plot Regression**: Fits a linear regression line to the log-quantile-quantile plot. A slope close to 1 and intercept close to 0 denote good calibration.

**Example Usage**:

.. code-block:: python

   import numpy as np
   from covtest.diagnostics.evaluate_pvalues import analyze_pvalues

   # Generate 1000 p-values: 950 under the null (Uniform) and 50 with true signal
   rng = np.random.default_rng(42)
   p_null = rng.uniform(0, 1, size=950)
   p_sig = rng.beta(0.1, 10, size=50)  # skewed towards 0
   p_values = np.concatenate([p_null, p_sig])

   # Analyze the p-value distribution
   results = analyze_pvalues(p_values)
   print("KS uniformity p-value:", results['ks']['pval'])
   print("Genomic Inflation Factor (lambda_GC):", results['inflation_factor'])
   print("Storey's pi0 estimate:", results['storey_pi0'])
   print("Binomial enrichment at alpha=0.01 p-value:", results['tail_tests'][0.01]['binom_p'])
   print("QQ plot regression slope:", results['qq_fit']['slope'])
