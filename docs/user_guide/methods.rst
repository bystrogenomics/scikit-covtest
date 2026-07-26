.. _methods:

Covariance Testing Methods
==========================

The primary focus of ``scikit-covtest`` is to provide statistically principled, high-dimensional hypothesis tests for covariance matrices. These are grouped into four main submodules:

1. **Identity Covariance Tests**: Test if a covariance matrix is equal to a specified matrix (typically the identity matrix).
2. **Sphericity Covariance Tests**: Test if a covariance matrix is proportional to the identity matrix.
3. **Two-Sample Equality Tests**: Test if two groups have the same covariance matrix.
4. **Proportionality Tests**: Test if two (or more) covariance matrices are proportional to each other.

All testing methods are implemented in the :mod:`covtest.methods` submodule.

One-Sample Identity Tests
-------------------------

Identity tests evaluate the null hypothesis :math:`H_0: \Sigma = I_p` (or more generally, :math:`H_0: \Sigma = \Sigma_0` by whitening the data beforehand). They are implemented in :mod:`covtest.methods.hypothesis_identity`:

- **Fisher's test** (``fisher_single_sample``): A high-dimensional test based on a fourth-order U-statistic estimator.
- **Srivastava (2011) test** (``srivastava2011_single_sample``): A high-dimensional test that is robust under non-normal distributions.
- **Srivastava (2005) test** (``srivastava_2005_identity``): An asymptotic normal test statistic.
- **Ledoit-Wolf test** (``ledoit_wolf_identity``): A classic test based on the trace of the squared difference between the sample covariance and the identity.
- **Nagao's test** (``nagao_identity``): A trace-based test using quadratic forms.
- **Tyler's M-estimator test** (``tyler_identity``): A distribution-free robust test using Tyler's shape matrix.
- **T2 trace estimator test** (``test_identity_T2``): A trace-based test optimized for large $p$ relative to $n$.
- **One-sample Likelihood Ratio Test** (``one_sample_cov_test``): A likelihood ratio test with asymptotic calibration.

**Example Usage**:

.. code-block:: python

   import numpy as np
   from covtest.methods.hypothesis_identity import fisher_single_sample

   # Generate normal data under H0 (n=50, p=10)
   rng = np.random.default_rng(42)
   X = rng.normal(size=(50, 10))

   # Run Fisher's identity test
   result = fisher_single_sample(X, Sigma="identity")
   print("Fisher's test statistic:", result['stat'])
   print("Fisher's test p-value:", result['p_value'])

One-Sample Sphericity Tests
---------------------------

Sphericity tests evaluate the null hypothesis :math:`H_0: \Sigma = \sigma^2 I_p` for some unknown scalar :math:`\sigma^2 > 0`. They are implemented in :mod:`covtest.methods.hypothesis_spherical`:

- **Srivastava (2005) test** (``srivastava_2005_sphericity``): A high-dimensional sphericity test under normal assumptions.
- **Srivastava-Yanagihara-Kubokawa (2014) test** (``srivastava_2014_sphericity_test``): A distribution-free test robust to non-normal distributions.
- **John's sphericity test** (``john_sphericity``): A classic sphericity test based on the ratio of trace estimators.
- **Chen-Zhang-Zhong (2010) test** (``czz_sphericity_test``): A location-invariant high-dimensional U-statistic sphericity test.
- **Ahmad (2015) sphericity test** (``ahmad2015_sphericity_test``): A high-dimensional trace ratio test.
- **Bartlett's sphericity test** (``bartlett_sphericity_test``): A classic likelihood ratio test (requires $p < n$).
- **Hu-Li-Liu-Zhou (2019) tests** (``hu_2019_sphericity_test``): Spatial-sign-based sphericity tests for elliptical populations.
- **Muirhead's Likelihood Ratio Test** (``muirhead_sphericity_lrt``): LRT for sphericity with small-sample Bartlett correction.

**Example Usage**:

.. code-block:: python

   import numpy as np
   from covtest.methods.hypothesis_spherical import srivastava_2005_sphericity

   # Generate spherical data with non-unit variance (n=60, p=20, sigma=2)
   rng = np.random.default_rng(42)
   X = rng.normal(scale=2.0, size=(60, 20))

   # Run Srivastava (2005) sphericity test
   result = srivastava_2005_sphericity(X)
   print("Sphericity test statistic:", result['stat'])
   print("Sphericity test p-value:", result['p_value'])

Two-Sample Equality Tests
--------------------------

Two-sample tests evaluate the null hypothesis that two groups share the same covariance matrix: :math:`H_0: \Sigma_1 = \Sigma_2`. They are implemented in :mod:`covtest.methods.hypothesis_two_sample`:

- **Schott (2001) test** (``schott_2001``): A high-dimensional test based on the trace of the squared difference of sample covariances.
- **Srivastava (2007) test** (``srivastava_two_sample_2007``): High-dimensional chi-square test statistic.
- **Srivastava-Yanagihara (2010) test** (``srivastava_yanagihara_two_sample``): Robust to non-normality.
- **Ahmad (2017) test** (``ahmad_2017_two_sample``): Uses Gram-matrix based trace estimators for high-dimensional settings.
- **Box's M test** (``boxm_test``): Classic multivariate test using either chi-squared or F reference distributions (requires $p < n$).
- **Wald test** (``wald_two_sample``): Classic two-sample Wald test (requires $p < n$).
- **Tyler's robust shape test** (``tyler_two_sample``): Distribution-free shape matrix equality test using Tyler's M-estimator.
- **Cai (2013) test** (``cai_2013_two_sample``): Extreme-value-based maximum difference test.
- **Ishii test** (``ishii_two_sample``): Uses noise-reduction PCA (designed for extremely high dimensions $p \gg n$).

**Example Usage**:

.. code-block:: python

   import numpy as np
   from covtest.methods.hypothesis_two_sample import schott_2001

   rng = np.random.default_rng(42)
   # Group 1 (n=40, p=15)
   X = rng.normal(size=(40, 15))
   # Group 2 (n=50, p=15)
   Y = rng.normal(size=(50, 15))

   # Run Schott (2001) equality test
   result = schott_2001(X, Y)
   print("Equality test statistic:", result['stat'])
   print("Equality test p-value:", result['p_value'])

Proportionality Tests
---------------------

Proportionality tests evaluate the null hypothesis :math:`H_0: \Sigma_1 = c \Sigma_2` for some scalar :math:`c > 0`. They are implemented in :mod:`covtest.methods.hypothesis_proportionality`:

- **Bartlett-adjusted test** (``bartlett_adjusted_proportionality_test``): Bartlett-corrected Wilks LRT utilizing parametric bootstrap.
- **Liu-Xu-Zheng-Tian (2014) test** (``proportionality_test_LZ``): High-dimensional proportionality test (requires $p < n_2$).
- **Cheng robust test** (``proportionality_test_signs``): Robust high-dimensional test based on spatial sign covariance. Highly suitable for $p > n$ and heavy-tailed data.
- **Pseudo-Likelihood Ratio Test (PLRT)** (``proportionality_plrt``): Pseudo-LRT utilizing random matrix theory limits.
- **Tsukuda trace-based test** (``proportional_cov_test_tsukuda``): Trace-based test for high-dimensional settings.
- **Ahmad (2022) test** (``ahmad_2022_proportionality_test``): High-dimensional trace-based test using $U$-statistics.

**Example Usage**:

.. code-block:: python

   import numpy as np
   from covtest.methods.hypothesis_proportionality import bartlett_adjusted_proportionality_test

   rng = np.random.default_rng(42)
   # Group 1 (n=30, p=5)
   X = rng.normal(scale=1.0, size=(30, 5))
   # Group 2 (n=30, p=5) proportional to Group 1
   Y = rng.normal(scale=2.0, size=(30, 5))

   # Run Bartlett-adjusted proportionality test
   result = bartlett_adjusted_proportionality_test(X, Y, B=100, random_state=42)
   print("Adjusted test statistic:", result['stat'])
   print("Adjusted p-value:", result['p_value'])
