# Tutorial: Identity Tests for Covariance Matrices
# =================================================
#
# This tutorial demonstrates how to test whether a covariance matrix
# equals the identity matrix: H0: Σ = I_p
#
# Identity tests are fundamental in multivariate analysis and arise in:
# - Checking if data has been properly standardized
# - Testing independence after whitening transformations
# - Model diagnostics in factor analysis

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from covtest.methods import (
     nagao_identity,
     ledoit_wolf_identity,
     srivastava_2005_identity,
     fisher_single_sample,
     ahmad2015_identity,
)
from covtest.datasets import load_mnist, load_iris

# ============================================================================
# Example 1: Basic Usage with Simulated Data
# ============================================================================

# Use Generator for reproducible random numbers (recommended over np.random.seed)
rng = np.random.default_rng(42)

# Generate data from N(0, I) - null hypothesis is TRUE
n, p = 100, 20
X_identity = rng.standard_normal((n, p))

# Generate data from N(0, Σ) where Σ ≠ I - null hypothesis is FALSE
Sigma = np.eye(p)
Sigma[0, 1] = Sigma[1, 0] = 0.5  # Add correlation
X_correlated = rng.multivariate_normal(np.zeros(p), Sigma, size=n)

print("=" * 60)
print("Example 1: Testing H0: Σ = I with simulated data")
print("=" * 60)

# Compute sample covariance to visualize
S_identity = np.cov(X_identity.T)
S_correlated = np.cov(X_correlated.T)

print(f"\nData from identity covariance:")
print(f"  Sample variance of first variable: {S_identity[0,0]:.3f} (expected: 1)")
print(f"  Sample correlation (1,2): {S_identity[0,1]/np.sqrt(S_identity[0,0]*S_identity[1,1]):.3f} (expected: 0)")

print(f"\nData from non-identity covariance:")
print(f"  Sample variance of first variable: {S_correlated[0,0]:.3f}")
print(f"  Sample correlation (1,2): {S_correlated[0,1]/np.sqrt(S_correlated[0,0]*S_correlated[1,1]):.3f} (expected: 0.5)")

# ============================================================================
# Example 2: Applying Identity Tests
# ============================================================================

print("\n" + "=" * 60)
print("Example 2: Applying identity tests")
print("=" * 60)

# Apply tests to both datasets
print("\n--- Testing data generated from identity covariance ---")
result = nagao_identity(X_identity)
print(f"Nagao test:       stat = {result['stat']:.2f}, p-value = {result['p_value']:.4f}")
result = ledoit_wolf_identity(X_identity)
print(f"Ledoit-Wolf test: stat = {result['stat']:.2f}, p-value = {result['p_value']:.4f}")

print("\n--- Testing data with non-identity covariance ---")
result = nagao_identity(X_correlated)
print(f"Nagao test:       stat = {result['stat']:.2f}, p-value = {result['p_value']:.4f}")
result = ledoit_wolf_identity(X_correlated)
print(f"Ledoit-Wolf test: stat = {result['stat']:.2f}, p-value = {result['p_value']:.4f}")

# ============================================================================
# Example 3: High-Dimensional Setting (p comparable to n)
# ============================================================================

print("\n" + "=" * 60)
print("Example 3: High-dimensional setting (p/n → c)")
print("=" * 60)

# When p is large relative to n, use high-dimensional tests
n_hd, p_hd = 50, 40  # p/n = 0.8
rng_hd = np.random.default_rng(123)
X_hd_null = rng_hd.standard_normal((n_hd, p_hd))

# Create alternative with spiked covariance
spike = rng_hd.standard_normal(p_hd)
spike = spike / np.linalg.norm(spike)
Sigma_spiked = np.eye(p_hd) + 2 * np.outer(spike, spike)
X_hd_alt = rng_hd.multivariate_normal(np.zeros(p_hd), Sigma_spiked, size=n_hd)

print(f"\nDimension ratio p/n = {p_hd/n_hd:.2f}")
print("\nFor high-dimensional data, prefer:")
print("  - Ledoit-Wolf (2002)")
print("  - Srivastava (2005)")  
print("  - Ahmad & von Rosen (2015)")

# ============================================================================
# Example 4: Real Data - Testing Standardized MNIST Digits
# ============================================================================

print("\n" + "=" * 60)
print("Example 4: MNIST application")
print("=" * 60)

print("""
Application: After standardizing MNIST digit images, we can test
whether the resulting covariance is identity.

If H0 is rejected, it suggests:
- Residual correlations between pixels
- The standardization didn't fully decorrelate the data
- Potential for dimensionality reduction (PCA)

Code:
    from covtest.datasets import load_mnist
    from covtest.methods import srivastava_2005_identity
    
    X, y = load_mnist(split="train")
    X_digit0 = X[y == 0][:500]  # First 500 zeros
    
    # Standardize
    X_std = (X_digit0 - X_digit0.mean(axis=0)) / X_digit0.std(axis=0)
    
    # Test identity
    result = srivastava_2005_identity(X_std)
    print(f"p-value: {result['p_value']:.4f}")
""")

# ============================================================================
# Choosing the Right Test
# ============================================================================

print("\n" + "=" * 60)
print("Guide: Choosing the Right Identity Test")
print("=" * 60)

print("""
| Scenario                    | Recommended Test              |
|-----------------------------|-------------------------------|
| Classical (n >> p)          | Nagao (1973)                  |
| High-dimensional (p ~ n)    | Ledoit-Wolf (2002)            |
|                             | Srivastava (2005)             |
| Very high-dim (p > n)       | Ahmad & von Rosen (2015)      |
| Need robustness             | Li et al. (2025)              |
| General purpose             | Fisher (2012)                 |

All tests in scikit-covtest return: {"stat": float, "p_value": float}
""")
