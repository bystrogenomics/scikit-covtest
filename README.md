# scikit-covtest: Covariance Matrix Hypothesis Testing Toolkit

`scikit-covtest` is a Python package designed for hypothesis testing on covariance matrices. It provides a comprehensive suite of statistical tests for high-dimensional data, including one-sample tests (Identity, Sphericity), two-sample tests (Equality), and multi-sample tests (Proportionality).

## Features

- **One-Sample Tests**:
    - **Identity**: Test if the covariance matrix is the identity matrix ($\Sigma = I$).
    - **Sphericity**: Test if the covariance matrix is proportional to the identity matrix ($\Sigma = \lambda I$).
- **Two-Sample Tests**:
    - **Equality**: Test if two covariance matrices are equal ($\Sigma_1 = \Sigma_2$).
- **Multi-Sample Tests**:
    - **Proportionality**: Test if multiple covariance matrices are proportional to each other ($\Sigma_i = c_i \Sigma$).
- **High-Dimensional Support**: Many tests are designed to work well even when the number of features ($p$) exceeds the number of samples ($n$).
- **Scikit-Learn Compatible**: Designed to integrate smoothly with the scientific Python ecosystem.

## Installation

You can install `scikit-covtest` directly from the source:

```bash
git clone https://github.com/bystrogenomics/scikit-covtest.git
cd scikit-covtest
pip install .
```

## Usage Examples

### 1. One-Sample Test: Identity

Test if the covariance matrix of a dataset is the Identity matrix.

```python
import numpy as np
from covtest.methods.hypothesis_identity import fisher_single_sample

# Generate synthetic data (Identity covariance)
rng = np.random.default_rng(42)
X = rng.normal(size=(50, 10))  # n=50, p=10

# Perform Fisher's test for Identity
result = fisher_single_sample(X, Sigma="identity")
print(f"Statistic: {result['stat']:.4f}, P-value: {result['p_value']:.4f}")
```

### 2. Two-Sample Test: Equality

Test if two datasets share the same covariance matrix.

```python
import numpy as np
from covtest.methods.hypothesis_two_sample import schott_2001

# Generate two datasets with the same covariance
rng = np.random.default_rng(42)
X1 = rng.normal(size=(30, 5))
X2 = rng.normal(size=(40, 5))

# Perform Schott's test for equality of covariance matrices
result = schott_2001(X1, X2)
print(f"Statistic: {result['stat']:.4f}, P-value: {result['p_value']:.4f}")
```

### 3. Multi-Sample Test: Proportionality

Test if covariance matrices from multiple groups are proportional.

```python
import numpy as np
from covtest.methods.hypothesis_proportionality import bartlett_adjusted_proportionality_test

# Generate data for 2 groups with proportional covariances
rng = np.random.default_rng(42)
cov = np.eye(5)
X1 = rng.multivariate_normal(mean=np.zeros(5), cov=cov, size=30)
X2 = rng.multivariate_normal(mean=np.zeros(5), cov=2*cov, size=30) # Proportional by factor 2

# Perform Bartlett-adjusted test for proportionality
result = bartlett_adjusted_proportionality_test(X1, X2)
print(f"Statistic: {result['stat']:.4f}, P-value: {result['p_value']:.4f}")
```

## Dependencies

- `numpy`
- `scipy`
- `scikit-learn`
- `pandas`
- `tqdm`
- `matplotlib`
- `statsmodels`
- `cvxpy`

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
