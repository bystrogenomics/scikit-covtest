# Spec: Fix degrees-of-freedom bugs in `hypothesis_identity.py`

## Context

Three functions in `hypothesis_identity.py` use `data.shape[0]` (the sample size N) where the referenced papers require `N - 1` (the degrees of freedom of the sample covariance matrix S). The `covariance_traces` and `sample_covariance` utilities in this codebase compute S with divisor `N - 1`, so every formula that divides by the paper's "n" must use `N - 1`, not `N`.

All three bugs inflate the test statistic by a factor of N/(N-1), producing anti-conservative p-values.

---

## Bug 1 of 3: `_ledoit_wolf_stat`

**Paper:** Ledoit & Wolf (2002), Equation (2). The paper defines n = number_of_observations - 1.

### Current code (broken)

```python
def _ledoit_wolf_stat(data):
    n, p = data.shape
    sample_cov_matrix, trace_S, _ = covariance_traces(data)
    SmI = sample_cov_matrix - np.eye(p)
    trace_smi2 = np.trace(SmI @ SmI)
    W = 1 / p * trace_smi2 - 1 / (n * p) * trace_S**2 + p / n
    return W
```

### Fixed code

```python
def _ledoit_wolf_stat(data):
    N, p = data.shape
    n = N - 1  # degrees of freedom (Ledoit & Wolf 2002, Assumption 2)
    sample_cov_matrix, trace_S, _ = covariance_traces(data)
    SmI = sample_cov_matrix - np.eye(p)
    trace_smi2 = np.trace(SmI @ SmI)
    W = 1 / p * trace_smi2 - 1 / (n * p) * trace_S**2 + p / n
    return W
```

### What changed

Line `n, p = data.shape` becomes `N, p = data.shape` then `n = N - 1`. Everything below stays the same.

---

## Bug 2 of 3: `ledoit_wolf_identity`

**Paper:** Ledoit & Wolf (2002), Proposition 6, eq. 18. Test statistic is `(n * p / 2) * W` where `n = N - 1`.

### Current code (broken)

```python
def ledoit_wolf_identity(X):
    X = validate_data_matrix(X)
    n, p = X.shape
    W = _ledoit_wolf_stat(X)
    degree_of_freedom = p * (p + 1) / 2
    stat = n * p / 2 * W
    p_value = 1 - stats.chi2.cdf(stat, degree_of_freedom)
    return result_dict(stat, p_value)
```

### Fixed code

```python
def ledoit_wolf_identity(X):
    X = validate_data_matrix(X)
    N, p = X.shape
    n = N - 1  # degrees of freedom
    W = _ledoit_wolf_stat(X)
    degree_of_freedom = p * (p + 1) / 2
    stat = n * p / 2 * W
    p_value = 1 - stats.chi2.cdf(stat, degree_of_freedom)
    return result_dict(stat, p_value)
```

### What changed

Line `n, p = X.shape` becomes `N, p = X.shape` then `n = N - 1`. Everything below stays the same.

---

## Bug 3 of 3: `nagao_identity`

**Paper:** Nagao (1973); Ledoit & Wolf (2002), eq. 13-14. Test statistic is `(n * p / 2) * V` where `n = N - 1`.

### Current code (broken)

```python
def nagao_identity(X):
    X = validate_data_matrix(X)
    n, p = X.shape
    V = _nagao_stat(X)
    degree_of_freedom = p * (p + 1) / 2
    stat = n * p / 2 * V
    p_value = 1 - stats.chi2.cdf(stat, degree_of_freedom)
    return result_dict(stat, p_value)
```

### Fixed code

```python
def nagao_identity(X):
    X = validate_data_matrix(X)
    N, p = X.shape
    n = N - 1  # degrees of freedom
    V = _nagao_stat(X)
    degree_of_freedom = p * (p + 1) / 2
    stat = n * p / 2 * V
    p_value = 1 - stats.chi2.cdf(stat, degree_of_freedom)
    return result_dict(stat, p_value)
```

### What changed

Line `n, p = X.shape` becomes `N, p = X.shape` then `n = N - 1`. Everything below stays the same.

---

## Do NOT modify these functions

They already use the correct degrees of freedom:

- `_nagao_stat` -- does not use n; leave as-is
- `srivastava_2005_identity` -- already passes `N - 1`
- `srivastava2011_single_sample` -- already passes `n - 1`
- `srivastava_2014_identity` -- already uses `n = N - 1`
- `fisher_single_sample` -- already uses `n_eff = N - 1`
- `test_identity_T2` (Ahmad 2015) -- correctly uses sample size n (U-statistic convention)
- `chen_2010_identity` -- correctly uses sample size n (U-statistic convention)
- `xu_2023_identity` -- correctly uses sample size n (U-statistic convention)
- `_srivastava2011_` -- receives n as a parameter; callers are already correct
- `_fisher_2012_stat_` -- receives n as a parameter; caller is already correct

---

## Verification

After applying the three fixes, run this test. The Ledoit-Wolf rejection rate at 5% should be approximately 5% (it was ~6.75% before the fix).

```python
import numpy as np
from scipy.stats import chi2

np.random.seed(999)
N, p, n_sim = 30, 8, 50000
df = p * (p + 1) / 2
rejects = 0
for _ in range(n_sim):
    X = np.random.normal(0, 1, (N, p))
    Xc = X - X.mean(axis=0)
    S = Xc.T @ Xc / (N - 1)
    trS = np.trace(S)
    SmI = S - np.eye(p)
    n = N - 1  # THE FIX
    W = (1/p) * np.trace(SmI @ SmI) - (1/(n*p)) * trS**2 + p/n
    stat = n * p / 2 * W
    if stat > chi2.ppf(0.95, df):
        rejects += 1
rate = rejects / n_sim
assert 0.04 <= rate <= 0.065, f"Rejection rate {rate:.4f} outside [0.04, 0.065]"
print(f"PASS: rejection rate = {rate:.4f}")
```
