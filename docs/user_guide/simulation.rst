.. _simulation:

Simulation and Synthetic Data Generation
========================================

``scikit-covtest`` includes a rich suite of tools for generating synthetic covariance matrices and simulating multivariate datasets. These are highly useful for benchmarking new covariance estimators, stress-testing hypothesis tests, and evaluating statistical power under controlled null and alternative settings.

The simulation functionality is split into two modules:

1. **Covariance Matrix Generators**: Generating structured, symmetric positive-definite matrices.
2. **Data Generators**: Simulating multivariate samples from Gaussian, heavy-tailed, or non-Gaussian distributions matching a target covariance structure.

All simulation tools are implemented in the :mod:`covtest.simulation` submodule.

Covariance Matrix Generators
----------------------------

The :mod:`covtest.simulation.generate_covariances` module provides several functions to generate covariance matrices with different structural properties:

- **Spiked Covariance Model** (``generate_spiked_covariance``): Generates a covariance matrix with one or more large "spiked" eigenvalues above a baseline noise variance.
- **Toeplitz / AR(1) Structure** (``generate_toeplitz_cov``): Simulates exponentially decaying correlations similar to an autoregressive model.
- **Block-Diagonal Structure** (``generate_block_diagonal_cov``): Simulates independent groups/blocks of correlated features.
- **Spectral Decomposition** (``generate_spectral_cov``): Samples eigenvalues uniformly and constructs a covariance matrix using a random orthogonal basis.
- **Low-Rank + Noise** (``generate_low_rank_cov``): Constructs a covariance matrix representing a factor model with latent factors plus isotropic noise.
- **Sparse Precision Model** (``generate_sparse_precision_cov``): Constructs a sparse precision (inverse covariance) matrix and inverts it.
- **Wishart / Marčenko–Pastur Ensemble** (``generate_marchenko_pastur``): Computes the sample covariance matrix of standard i.i.d. normal samples.
- **Gaussian Orthogonal Ensemble (GOE)** (``sample_goe``): Samples a symmetric matrix from the GOE.
- **Uniform Correlation** (``sample_cov_uniform_correlation``): Constructs a matrix with unit variances and off-diagonal correlations drawn uniformly from a given range (low, medium, or high).

**Example Usage**:

.. code-block:: python

   import numpy as np
   from covtest.simulation.generate_covariances import generate_toeplitz_cov, generate_spiked_covariance

   # Generate a Toeplitz matrix with AR(1) correlation rho=0.7
   Sigma_toeplitz = generate_toeplitz_cov(p=10, rho=0.7)
   print("Toeplitz Covariance shape:", Sigma_toeplitz.shape)

   # Generate a spiked covariance model with 2 spiked eigenvalues of 10.0
   Sigma_spiked = generate_spiked_covariance(p=10, spike_eigenvalue=10.0, num_spikes=2)
   print("Spiked eigenvalues:", np.linalg.eigvalsh(Sigma_spiked)[-4:])

Data Generators & Alternative Hypotheses
-----------------------------------------

To evaluate covariance testing procedures under realistic conditions, the :mod:`covtest.simulation.generate_data` module allows generating multivariate datasets with a target covariance matrix. Crucially, the marginal distributions can be configured to represent heavy-tailed or non-Gaussian laws:

- **Heavy-Tailed Data Generation** (``generate_heavy_tailed_samples``): Generates samples matching the target covariance matrix via Cholesky decomposition. Supported distributions (``dist_type``) include:

  * ``'normal'``: Standard multivariate normal.
  * ``'t'``: Student's t-distribution (requires a degrees of freedom parameter ``df``).
  * ``'laplace'``: Laplace distribution (double exponential).
  * ``'lognormal'``: Log-normal distribution (requires a shape parameter ``s``).
  * ``'pareto'``: Symmetric Pareto (requires a shape parameter ``b``).
  * ``'scale_mixture'``: Gaussian scale mixture.
  * ``'variance_gamma'``: Gamma-normal mixture.
  * ``'gh'``: Generalized hyperbolic approximation.

- **Simulation Under Alternatives** (``generate_heavy_tailed_alternative``): Simulates datasets under specific alternative covariance models on top of a baseline heavy-tailed noise. Supported alternatives (passed in the ``options`` dictionary under ``'alt'``) include:

  * ``'mixture'``: A two-component mixture model (adds $\pm \mu$ to the samples).
  * ``'scaled_cov'``: Multiplies the covariance matrix by a scale factor.
  * ``'rank1_bump'``: Adds a rank-1 perturbation ($\Sigma + v v^T$).
  * ``'eig_bump'``: Adds a perturbation to the leading mode (largest eigenvector).

**Example Usage**:

.. code-block:: python

   import numpy as np
   from covtest.simulation.generate_covariances import generate_toeplitz_cov
   from covtest.simulation.generate_data import generate_heavy_tailed_samples, generate_heavy_tailed_alternative

   rng = np.random.default_rng(42)
   # Define a target covariance matrix
   Sigma = generate_toeplitz_cov(p=5, rho=0.5)

   # 1. Generate heavy-tailed samples from a Student's t distribution with df=4
   X_t = generate_heavy_tailed_samples(cov=Sigma, n=100, dist_type="t", rng=rng, options={"df": 4})
   print("Student-t samples shape:", X_t.shape)

   # 2. Generate samples under a rank-1 bump alternative model
   # options specify a rank-1 perturbation of norm 2.0 and a Student-t noise with df=5
   opt = {
       "alt": "rank1_bump",
       "v_norm": 2.0,
       "dist": {"df": 5}
   }
   res = generate_heavy_tailed_alternative(cov=Sigma, n=100, dist_type="t", rng=rng, options=opt)
   print("Generated alternative data shape:", res['X'].shape)
   print("Perturbed covariance used:\n", res['cov_used'])
