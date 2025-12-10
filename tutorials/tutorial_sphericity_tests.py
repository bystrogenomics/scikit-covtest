# Tutorial: Sphericity Tests for Covariance Matrices
# ===================================================
#
# This tutorial demonstrates how to test whether a covariance matrix
# is proportional to the identity: H0: Σ = σ² I_p (for some σ² > 0)
#
# Sphericity tests are used in:
# - Testing assumptions for repeated measures ANOVA (Mauchly's test)
# - Checking isotropy in spatial statistics
# - Validating PCA assumptions
# - Testing for equal variances with zero correlations

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# For actual usage:
# from covtest.methods import (
#     bartlett_sphericity_test,
#     john_sphericity,
#     srivastava_2005_sphericity,
#     hallin_rank_sphericity_test,
#     czz_sphericity_test,
#     muirhead_sphericity_lrt,
# )
# from covtest.datasets import load_iris

# ============================================================================
# Example 1: Understanding Sphericity
# ============================================================================

# Use Generator for reproducible random numbers
rng = np.random.default_rng(42)

print("=" * 60)
print("Example 1: What is Sphericity?")
print("=" * 60)

print("""
Sphericity means: Σ = σ² I_p

This implies:
1. All variables have equal variance (σ²)
2. All pairs of variables are uncorrelated

Spherical data forms a "hypersphere" in p-dimensional space.
Non-spherical data forms an "ellipsoid".
""")

# Generate spherical data (H0 true)
n, p = 100, 4
sigma2 = 2.5  # Common variance
X_spherical = np.sqrt(sigma2) * rng.standard_normal((n, p))

# Generate non-spherical data (H0 false) - different variances
variances = [1, 2, 3, 4]
X_nonspherical = np.column_stack([
    np.sqrt(v) * rng.standard_normal(n) for v in variances
])

print("Spherical data (equal variances, no correlation):")
S1 = np.cov(X_spherical.T)
print(f"  Variances: {np.diag(S1).round(2)}")
print(f"  Expected:  [{sigma2:.1f}, {sigma2:.1f}, {sigma2:.1f}, {sigma2:.1f}]")

print("\nNon-spherical data (unequal variances):")
S2 = np.cov(X_nonspherical.T)
print(f"  Variances: {np.diag(S2).round(2)}")
print(f"  Expected:  [1, 2, 3, 4]")

# ============================================================================
# Example 2: Applying Sphericity Tests
# ============================================================================

print("\n" + "=" * 60)
print("Example 2: Applying sphericity tests")
print("=" * 60)

def demo_john_sphericity(X):
    """John (1971) test for sphericity"""
    n, p = X.shape
    S = np.cov(X.T)
    trace_S = np.trace(S)
    trace_S2 = np.trace(S @ S)
    U = (1/p) * trace_S2 / ((1/p) * trace_S)**2 - 1
    stat = U * n * p / 2
    df = p * (p + 1) / 2 - 1
    p_value = 1 - stats.chi2.cdf(stat, df)
    return {"stat": stat, "p_value": p_value}

def demo_bartlett_sphericity(X):
    """Bartlett's test - tests correlation matrix = I"""
    n, p = X.shape
    R = np.corrcoef(X.T)
    sign, logdet = np.linalg.slogdet(R)
    stat = -(n - 1 - (2*p + 5)/6) * logdet
    df = p * (p - 1) / 2
    p_value = stats.chi2.sf(stat, df)
    return {"stat": stat, "p_value": p_value}

print("\n--- Testing spherical data ---")
result = demo_john_sphericity(X_spherical)
print(f"John's test:     stat = {result['stat']:.2f}, p-value = {result['p_value']:.4f}")
result = demo_bartlett_sphericity(X_spherical)
print(f"Bartlett's test: stat = {result['stat']:.2f}, p-value = {result['p_value']:.4f}")

print("\n--- Testing non-spherical data ---")
result = demo_john_sphericity(X_nonspherical)
print(f"John's test:     stat = {result['stat']:.2f}, p-value = {result['p_value']:.4f}")
result = demo_bartlett_sphericity(X_nonspherical)
print(f"Bartlett's test: stat = {result['stat']:.2f}, p-value = {result['p_value']:.4f}")

# ============================================================================
# Example 3: Iris Dataset - Testing Sphericity by Species
# ============================================================================

print("\n" + "=" * 60)
print("Example 3: Iris Dataset Application")
print("=" * 60)

print("""
Application: Test if measurements within each Iris species are spherical.

This tests whether:
- Sepal length, sepal width, petal length, petal width
  have equal variance and zero correlation within species.

Code:
    from covtest.datasets import load_iris
    from covtest.methods import john_sphericity, bartlett_sphericity_test
    
    X, y = load_iris()
    
    for species in [0, 1, 2]:  # setosa, versicolor, virginica
        X_species = X[y == species]
        
        # Standardize to remove mean differences
        X_std = (X_species - X_species.mean(axis=0))
        
        result = john_sphericity(X_std)
        print(f"Species {species}: p-value = {result['p_value']:.4f}")

Expected outcome: Likely reject H0 for all species, as petal
measurements typically have different variance than sepal measurements.
""")

# ============================================================================
# Example 4: High-Dimensional Sphericity (TCGA Gene Expression)
# ============================================================================

print("\n" + "=" * 60)
print("Example 4: High-dimensional sphericity (gene expression)")
print("=" * 60)

print("""
For high-dimensional data (p >> n), use:
- Srivastava (2005) sphericity test
- Chen-Zhang-Zhong (2010) test  
- Hallin & Paindaveine (2006) rank-based test (robust)

Application: TCGA gene expression data
    - Test if gene expression is isotropic within a tumor type
    - Rejection suggests structured covariance (gene modules, pathways)

Code:
    from covtest.datasets import load_tcga
    from covtest.methods import srivastava_2005_sphericity, czz_sphericity_test
    
    X, y = load_tcga()
    
    # Select one cancer type and subset of genes
    X_cancer = X[y == 0][:, :500]  # First 500 genes
    
    # High-dimensional test
    result = czz_sphericity_test(X_cancer)
    print(f"CZZ test p-value: {result['p_value']:.4f}")
""")

# ============================================================================
# Choosing the Right Sphericity Test
# ============================================================================

print("\n" + "=" * 60)
print("Guide: Choosing the Right Sphericity Test")
print("=" * 60)

print("""
| Scenario                    | Recommended Test              |
|-----------------------------|-------------------------------|
| Classical (n >> p)          | John (1971)                   |
|                             | Bartlett (1954)               |
|                             | Muirhead LRT                  |
| High-dimensional (p ~ n)    | Srivastava (2005)             |
|                             | Chen-Zhang-Zhong (2010)       |
| Need robustness             | Hallin-Paindaveine (2006)     |
| Heavy-tailed data           | SK test (Feng-Liu 2017)       |

Note: Bartlett's test specifically tests the CORRELATION matrix = I,
while other tests allow for unknown common variance σ².
""")
