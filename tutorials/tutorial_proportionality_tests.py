# Tutorial: Proportionality Tests for Covariance Matrices
# ========================================================
#
# This tutorial demonstrates how to test whether two covariance matrices
# are proportional: H0: Σ₁ = c·Σ₂ (for some c > 0)
#
# Proportionality tests are useful when:
# - Groups may have different overall variability but same correlation structure
# - Testing if variance scaling is the only difference between groups
# - Validating assumptions in discriminant analysis

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# For actual usage:
# from covtest.methods import (
#     flury_proportionality_test,
#     bartlett_adjusted_proportionality_test,
#     proportionality_test_LZ,
#     proportionality_test_signs,
#     proportional_cov_test_tsukuda,
# )
# from covtest.datasets import load_mnist, load_iris

# ============================================================================
# Example 1: Understanding Proportional Covariances
# ============================================================================

# Use Generator for reproducible random numbers
rng = np.random.default_rng(42)

print("=" * 60)
print("Example 1: What is Proportionality?")
print("=" * 60)

print("""
Proportionality means: Σ₁ = c · Σ₂

This implies:
- Same correlation structure in both groups
- Variances differ by a constant factor c across ALL variables
- The "shape" of the ellipsoid is the same, only "size" differs

This is WEAKER than equality (which requires c = 1).
""")

# Generate proportional covariances (H0 true)
n1, n2, p = 80, 100, 5

# Base covariance with correlations
Sigma_base = np.array([
    [1.0, 0.5, 0.3, 0.2, 0.1],
    [0.5, 1.0, 0.4, 0.3, 0.2],
    [0.3, 0.4, 1.0, 0.5, 0.3],
    [0.2, 0.3, 0.5, 1.0, 0.4],
    [0.1, 0.2, 0.3, 0.4, 1.0]
])

c = 2.0  # Proportionality constant
Sigma1 = Sigma_base
Sigma2 = c * Sigma_base  # Proportional

X = rng.multivariate_normal(np.zeros(p), Sigma1, size=n1)
Y = rng.multivariate_normal(np.zeros(p), Sigma2, size=n2)

S1 = np.cov(X.T)
S2 = np.cov(Y.T)

print(f"True proportionality constant c = {c}")
print(f"Estimated c (ratio of traces): {np.trace(S2)/np.trace(S1):.2f}")
print(f"\nCorrelation matrix similarity:")
R1 = np.corrcoef(X.T)
R2 = np.corrcoef(Y.T)
print(f"  Frobenius distance of correlation matrices: {np.linalg.norm(R1 - R2):.3f}")

# ============================================================================
# Example 2: Proportional vs Non-Proportional
# ============================================================================

print("\n" + "=" * 60)
print("Example 2: Detecting departure from proportionality")
print("=" * 60)

# Non-proportional: different correlation structure
Sigma3 = np.array([
    [2.0, 0.1, 0.1, 0.1, 0.1],  # Different correlations
    [0.1, 2.0, 0.1, 0.1, 0.1],
    [0.1, 0.1, 2.0, 0.1, 0.1],
    [0.1, 0.1, 0.1, 2.0, 0.1],
    [0.1, 0.1, 0.1, 0.1, 2.0]
])

Z = rng.multivariate_normal(np.zeros(p), Sigma3, size=n2)
S3 = np.cov(Z.T)

R3 = np.corrcoef(Z.T)
print("Non-proportional case (different correlation structure):")
print(f"  Frobenius distance of correlation matrices: {np.linalg.norm(R1 - R3):.3f}")

# ============================================================================
# Example 3: MNIST - Comparing Digit Covariances
# ============================================================================

print("\n" + "=" * 60)
print("Example 3: MNIST Application")
print("=" * 60)

print("""
Application: Are digit "1" and digit "7" covariances proportional?

If proportional: same pixel correlation patterns, different overall variance
If not proportional: fundamentally different spatial structure

Code:
    from covtest.datasets import load_mnist
    from covtest.methods import proportional_cov_test_tsukuda
    
    X, y = load_mnist(split="train")
    
    # Compare digits 1 and 7 (visually similar)
    X_1 = X[y == 1][:200]
    X_7 = X[y == 7][:200]
    
    # Use high-dimensional test (784 pixels, ~200 samples)
    result = proportional_cov_test_tsukuda(X_1, X_7)
    print(f"p-value: {result['p_value']:.4f}")
    
    # Compare with very different digits (0 vs 1)
    X_0 = X[y == 0][:200]
    result2 = proportional_cov_test_tsukuda(X_0, X_1)
    print(f"0 vs 1 p-value: {result2['p_value']:.4f}")
""")

# ============================================================================
# Example 4: Robust Proportionality Test
# ============================================================================

print("\n" + "=" * 60)
print("Example 4: Robust testing with spatial signs")
print("=" * 60)

print("""
For data with outliers or heavy tails, use:
    proportionality_test_signs(X, Y, calibration="permutation")

This test uses spatial signs instead of raw data, providing:
- Robustness to outliers
- Valid inference under elliptical distributions
- Works in high dimensions

Code:
    from covtest.methods import proportionality_test_signs
    
    # With outliers in the data
    result = proportionality_test_signs(
        X_contaminated, 
        Y_contaminated,
        center="spatial_median",  # Robust centering
        calibration="permutation",
        n_perm=999
    )
""")

# ============================================================================
# Choosing the Right Test
# ============================================================================

print("\n" + "=" * 60)
print("Guide: Choosing the Right Proportionality Test")
print("=" * 60)

print("""
| Scenario                    | Recommended Test              |
|-----------------------------|-------------------------------|
| Classical (n >> p)          | Flury (1986)                  |
|                             | Eriksen (1987)                |
| High-dimensional (p ~ n)    | Liu et al. (2014)             |
|                             | Tsukuda-Matsuura (2019)       |
| Need robustness             | Cheng et al. (2019) signs     |
| Small samples               | Eriksen with bootstrap        |

Note: Proportionality is a WEAKER hypothesis than equality.
If you reject proportionality, you would also reject equality.
""")
