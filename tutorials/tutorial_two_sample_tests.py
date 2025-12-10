# Tutorial: Two-Sample Tests for Covariance Matrix Equality
# ==========================================================
#
# This tutorial demonstrates how to test whether two groups have
# equal covariance matrices: H0: Σ₁ = Σ₂
#
# Two-sample covariance tests are essential for:
# - Validating homoscedasticity assumptions in MANOVA/discriminant analysis
# - Comparing variability structure across experimental conditions
# - Detecting differential co-expression in genomics

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# For actual usage:
# from covtest.methods import (
#     boxm_test,
#     schott2007,
#     srivastava_two_sample_2007,
#     srivastava_yanagihara_two_sample,
#     tyler_two_sample,
#     cai_2013_two_sample,
# )
# from covtest.datasets import load_mnist, load_iris, load_tcga

# ============================================================================
# Example 1: Basic Two-Sample Test
# ============================================================================

# Use Generator for reproducible random numbers
rng = np.random.default_rng(42)

print("=" * 60)
print("Example 1: Testing H0: Σ₁ = Σ₂")
print("=" * 60)

n1, n2, p = 50, 60, 4

# Equal covariances (H0 true)
Sigma = np.array([
    [1.0, 0.5, 0.3, 0.2],
    [0.5, 1.0, 0.4, 0.3],
    [0.3, 0.4, 1.0, 0.5],
    [0.2, 0.3, 0.5, 1.0]
])

X_equal = rng.multivariate_normal(np.zeros(p), Sigma, size=n1)
Y_equal = rng.multivariate_normal(np.zeros(p), Sigma, size=n2)

# Unequal covariances (H0 false)
Sigma2 = np.array([
    [2.0, 0.2, 0.1, 0.1],
    [0.2, 2.0, 0.2, 0.1],
    [0.1, 0.2, 2.0, 0.2],
    [0.1, 0.1, 0.2, 2.0]
])

Y_unequal = rng.multivariate_normal(np.zeros(p), Sigma2, size=n2)

print("Sample covariance matrices:")
S1 = np.cov(X_equal.T)
S2_eq = np.cov(Y_equal.T)
S2_uneq = np.cov(Y_unequal.T)

print(f"\nEqual case - Frobenius distance: {np.linalg.norm(S1 - S2_eq):.3f}")
print(f"Unequal case - Frobenius distance: {np.linalg.norm(S1 - S2_uneq):.3f}")

# ============================================================================
# Example 2: Box's M Test (Classical)
# ============================================================================

print("\n" + "=" * 60)
print("Example 2: Box's M Test")
print("=" * 60)

def demo_boxm_test(X, Y):
    """Box's M test for equality of covariances"""
    n, p = X.shape
    m = Y.shape[0]
    
    s1 = np.cov(X, rowvar=False)
    s2 = np.cov(Y, rowvar=False)
    s_pooled = ((n-1)*s1 + (m-1)*s2) / (n + m - 2)
    
    log_M = ((n-1)*np.log(np.linalg.det(s1)) + 
             (m-1)*np.log(np.linalg.det(s2)) - 
             (n+m-2)*np.log(np.linalg.det(s_pooled))) / 2
    
    c1 = (1/(n-1) + 1/(m-1) - 1/(n+m-2)) * (2*p**2 + 3*p - 1) / (6*(p+1))
    stat = -2 * (1 - c1) * log_M
    df = p * (p + 1) / 2
    p_value = stats.chi2.sf(stat, df)
    
    return {"stat": stat, "p_value": p_value}

print("\n--- Equal covariances ---")
result = demo_boxm_test(X_equal, Y_equal)
print(f"Box's M: stat = {result['stat']:.2f}, p-value = {result['p_value']:.4f}")

print("\n--- Unequal covariances ---")
result = demo_boxm_test(X_equal, Y_unequal)
print(f"Box's M: stat = {result['stat']:.2f}, p-value = {result['p_value']:.4f}")

# ============================================================================
# Example 3: Iris Dataset - Comparing Species
# ============================================================================

print("\n" + "=" * 60)
print("Example 3: Iris Dataset - Comparing Species Covariances")
print("=" * 60)

print("""
Application: Do different Iris species have different covariance structures?

This is important for:
- Linear Discriminant Analysis assumes equal covariances
- Quadratic DA allows unequal covariances
- Test determines which method is appropriate

Code:
    from covtest.datasets import load_iris
    from covtest.methods import boxm_test
    
    X, y = load_iris()
    
    # Compare setosa vs versicolor
    X_setosa = X[y == 0]
    X_versicolor = X[y == 1]
    
    result = boxm_test(X_setosa, X_versicolor)
    print(f"Setosa vs Versicolor: p = {result['p_value']:.4f}")
    
    # Compare all pairs
    for i, j in [(0,1), (0,2), (1,2)]:
        result = boxm_test(X[y==i], X[y==j])
        print(f"Species {i} vs {j}: p = {result['p_value']:.4f}")
""")

# ============================================================================
# Example 4: High-Dimensional Two-Sample Tests
# ============================================================================

print("\n" + "=" * 60)
print("Example 4: High-Dimensional Tests (TCGA/MNIST)")
print("=" * 60)

print("""
When p is large relative to n, classical tests fail.
Use high-dimensional alternatives:

1. Schott (2007) - trace-based, works for p > n
2. Srivastava (2007) - works for p comparable to n  
3. Srivastava-Yanagihara (2010) - robust to high dimensions
4. Cai-Liu-Xia (2013) - max-norm based, good for sparse differences

Code for TCGA gene expression:
    from covtest.datasets import load_tcga
    from covtest.methods import schott2007, srivastava_two_sample_2007
    
    X, y = load_tcga()
    
    # Compare two cancer types
    X_cancer1 = X[y == 0]
    X_cancer2 = X[y == 1]
    
    # High-dimensional test
    result = schott2007(X_cancer1, X_cancer2)
    print(f"Schott 2007: p = {result['p_value']:.4f}")

Code for MNIST:
    from covtest.datasets import load_mnist
    from covtest.methods import schott2007
    
    X, y = load_mnist(split="train")
    
    # Compare digit classes
    X_0 = X[y == 0][:300]
    X_1 = X[y == 1][:300]
    
    result = schott2007(X_0, X_1)
    print(f"Digit 0 vs 1: p = {result['p_value']:.4f}")
""")

# ============================================================================
# Example 5: Robust Two-Sample Test with Tyler's M-estimator
# ============================================================================

print("\n" + "=" * 60)
print("Example 5: Robust Testing with Tyler's M-estimator")
print("=" * 60)

print("""
For data with outliers or heavy tails:
    tyler_two_sample(X, Y)

Tyler's M-estimator is:
- Distribution-free under elliptical distributions
- Robust to outliers (bounded influence)
- Tests shape matrix equality (invariant to scale)

Code:
    from covtest.methods import tyler_two_sample
    
    # Add some outliers
    X_contaminated = X.copy()
    X_contaminated[:5] *= 10  # Outliers
    
    # Standard test may be affected
    result_box = boxm_test(X_contaminated, Y)
    
    # Robust test handles outliers
    result_tyler = tyler_two_sample(X_contaminated, Y)
    print(f"Tyler's test: p = {result_tyler['p_value']:.4f}")
""")

# ============================================================================
# Choosing the Right Two-Sample Test
# ============================================================================

print("\n" + "=" * 60)
print("Guide: Choosing the Right Two-Sample Test")
print("=" * 60)

print("""
| Scenario                    | Recommended Test              |
|-----------------------------|-------------------------------|
| Classical (n >> p)          | Box's M (1953)                |
|                             | Wald test                     |
| High-dimensional (p ~ n)    | Schott (2007)                 |
|                             | Srivastava (2007)             |
|                             | Srivastava-Yanagihara (2010)  |
| Very high-dim (p > n)       | Schott (2007)                 |
|                             | Cai-Liu-Xia (2013)            |
| Sparse differences          | Cai-Liu-Xia (2013)            |
| Outliers/heavy tails        | Tyler two-sample              |
| Need robustness             | Ahmad (2017)                  |

Note: 
- Box's M is sensitive to non-normality
- High-dimensional tests don't require matrix inversion
- Tyler's test is invariant to the overall scale
""")

# ============================================================================
# Example 6: Multiple Comparisons
# ============================================================================

print("\n" + "=" * 60)
print("Example 6: Multiple Group Comparisons")
print("=" * 60)

print("""
When comparing k > 2 groups, consider:

1. Global test first: Test if ANY pair differs
   - Use k-sample extension of Box's M
   
2. Pairwise tests with correction:
   - Bonferroni: α/k(k-1)/2 for each pair
   - Holm's step-down procedure
   - FDR control (Benjamini-Hochberg)

Code:
    from covtest.methods import boxm_test
    from covtest.multiplicity import holm_correction
    
    # All pairwise tests
    p_values = []
    pairs = []
    for i in range(k):
        for j in range(i+1, k):
            result = boxm_test(X_groups[i], X_groups[j])
            p_values.append(result['p_value'])
            pairs.append((i, j))
    
    # Apply correction
    adjusted = holm_correction(p_values)
""")
