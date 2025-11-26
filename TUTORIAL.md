# scikit-covtest Tutorial

This tutorial guides you through the main features of `scikit-covtest`.

## 1. One-Sample Tests

### Identity Test
Test if the covariance matrix $\Sigma$ is equal to the Identity matrix $I$.

```python
import numpy as np
from covtest.methods.hypothesis_identity import fisher_single_sample

# Generate data (n=50, p=10) from N(0, I)
rng = np.random.default_rng(42)
X = rng.normal(size=(50, 10))

# Test H0: Sigma = I
result = fisher_single_sample(X, Sigma="identity")
print(f"Identity Test P-value: {result['p_value']:.4f}")
```

### Sphericity Test
Test if the covariance matrix $\Sigma$ is proportional to the Identity matrix ($\Sigma = \lambda I$).

```python
from covtest.methods.hypothesis_spherical import srivastava_2005_spherical

# Generate data from N(0, 2*I)
X_sphere = rng.normal(scale=np.sqrt(2), size=(50, 10))

# Test H0: Sigma = lambda * I
result = srivastava_2005_spherical(X_sphere)
print(f"Sphericity Test P-value: {result['p_value']:.4f}")
```

## 2. Two-Sample Tests

### Equality of Covariances
Test if two groups have the same covariance matrix ($\Sigma_1 = \Sigma_2$).

```python
from covtest.methods.hypothesis_two_sample import schott_2001

# Group 1: N(0, I)
X1 = rng.normal(size=(30, 5))
# Group 2: N(0, I)
X2 = rng.normal(size=(40, 5))

# Test H0: Sigma1 = Sigma2
result = schott_2001(X1, X2)
print(f"Equality Test P-value: {result['p_value']:.4f}")
```

## 3. Multi-Sample Tests

### Proportionality
Test if covariance matrices are proportional ($\Sigma_i = c_i \Sigma$).

```python
from covtest.methods.hypothesis_proportionality import bartlett_adjusted_proportionality_test

# Group 1: Cov = I
X1 = rng.normal(size=(30, 5))
# Group 2: Cov = 2*I
X2 = rng.normal(scale=np.sqrt(2), size=(30, 5))

# Test H0: Sigma1 = c * Sigma2
result = bartlett_adjusted_proportionality_test(X1, X2)
print(f"Proportionality Test P-value: {result['p_value']:.4f}")
```

## 4. High-Dimensional Data

`scikit-covtest` is designed for high-dimensional settings where $p > n$.

```python
# High-dimensional data (n=20, p=50)
X_hd = rng.normal(size=(20, 50))

# Fisher's identity test works even when p > n
result = fisher_single_sample(X_hd, Sigma="identity")
print(f"High-Dim Identity Test P-value: {result['p_value']:.4f}")
```
