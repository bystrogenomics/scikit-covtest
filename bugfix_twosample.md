# Bugfix Specification Sheet — `hypothesis_two_sample.py`

**Date:** 2025-07-13  
**Scope:** Three mathematical discrepancies confirmed by term-by-term comparison against the source papers and validated with Monte Carlo simulation.

---

## Bug 1 — Missing dimension factors in â₄ estimator (Srivastava & Yanagihara 2010)

### Severity: **HIGH**
### Affects
| Function | Lines (approx.) |
|---|---|
| `_srivastava_yanagihara_stat` | 393–400 |
| `_srivastava_2007_stat` | 535–543 |

Both functions feed into the public wrappers `srivastava_yanagihara_two_sample` and `srivastava_two_sample_2007`, so every call to either test produces an incorrect â₄, which propagates into the variance estimators ξ̂² / η̂² and ultimately into the test statistic and p-value.

### Paper reference
Srivastava, M. S. & Yanagihara, H. (2010). *Testing the equality of several covariance matrices with fewer observations than the dimension.* J. Multivariate Analysis, 101(6), 1319–1329, **Theorem 2.2, Eq. (2.1).**

### Correct formula (Theorem 2.2)

```
â₄ = (1/c₀) × { (1/m) tr V⁴
                − m  · c₁ · â₁
                − m² · c₂ · â₁² · â₂
                − m  · c₃ · â₂²
                − n  · m³ · â₁⁴ }
```

where `m` = number of features (`p` in the code), `n` = total pooled degrees of freedom (`ntot`), `V` = pooled cross-product matrix (`Apool`).

### Current code — `_srivastava_yanagihara_stat` (lines ~393-400)

```python
a4 = (1.0 / c0) * (
    np.trace(Apool @ Apool @ Apool @ Apool) / p
    - c1 * a1                        # ← missing × p
    - c2 * p * a1**2 * a2            # ← has p, should be p**2
    - c3 * a2**2                     # ← missing × p
    - ntot * p**3 * a1**4            # ✓ correct
)
```

### Fixed code — `_srivastava_yanagihara_stat`

```python
a4 = (1.0 / c0) * (
    np.trace(Apool @ Apool @ Apool @ Apool) / p
    - c1 * p * a1                    # FIXED: added × p
    - c2 * p**2 * a1**2 * a2         # FIXED: p → p**2
    - c3 * p * a2**2                 # FIXED: added × p
    - ntot * p**3 * a1**4
)
```

### Current code — `_srivastava_2007_stat` (lines ~535-543)

```python
a4 = (1.0 / c0) * (
    np.trace(Apool @ Apool @ Apool @ Apool) / p
    - c1 * a1                        # ← missing × p
    - c2 * a1**2 * a2                # ← missing × p**2
    - c3 * a2**2                     # ← missing × p
    - ntot * a1**4 * p**3            # ✓ correct
)
```

### Fixed code — `_srivastava_2007_stat`

```python
a4 = (1.0 / c0) * (
    np.trace(Apool @ Apool @ Apool @ Apool) / p
    - c1 * p * a1                    # FIXED: added × p
    - c2 * p**2 * a1**2 * a2         # FIXED: added × p**2
    - c3 * p * a2**2                 # FIXED: added × p
    - ntot * a1**4 * p**3
)
```

### Monte Carlo evidence (Σ = I, true a₄ = 1.0)

| Formula | E[â₄] (p=3, n=50) | E[â₄] (p=5, n=100) |
|---------|---------|---------|
| Paper (correct) | 0.989 | 0.995 |
| `_srivastava_yanagihara_stat` (current) | 1.232 | 1.240 |
| `_srivastava_2007_stat` (current) | 1.236 | 1.243 |

### Verification test

Generate V ~ W_m(I_m, n) with m=5, n=100 (50 000 reps). True a₄ = 1. Assert |E[â₄] − 1| < 0.05.

---

## Bug 2 — Incorrect h* formula and null distribution (Ishii 2017)

### Severity: **HIGH**
### Affects
| Function | Lines (approx.) |
|---|---|
| `ishii_two_sample` | 248–262 |

### Paper reference
Ishii, A. (2017). *A high-dimensional two-sample test for non-Gaussian data under a strongly spiked eigenvalue model.* J. Japan Statist. Soc., 47(2), 273–291, **Section 5 and Theorem 5.1.**

### Bug 2a — `h_star` formula

**Paper:** h* = max{h, 1/h}

**Current code (line ~250):**

```python
h_star = (h_dot + 1 / h_dot) / 2          # arithmetic mean — WRONG
```

**Fixed code:**

```python
h_star = max(h_dot, 1.0 / h_dot)          # paper's max definition
```

**Numerical impact:** h=2 → paper: 2.0, code: 1.25 (37.5% error)

### Bug 2b — Null distribution

**Paper (Theorem 5.1):** Under H₀, F₁ ⇒ {F(ν₁,ν₂) × F(ν₁,ν₂)}^{1/2}, where the two F variables are independent.

**Current code:** Uses a simple F(ν₁, ν₂) distribution.

```python
p_value = 2 * min(f.cdf(F3, nu1, nu2), 1 - f.cdf(F3, nu1, nu2))
```

**Fixed code (Monte Carlo null CDF):**

```python
from scipy.stats import f as f_dist

def _product_f_cdf(x, nu1, nu2, n_mc=200_000, seed=0):
    # CDF of sqrt(F1 * F2), F1 F2 iid F(nu1, nu2)
    rng = np.random.default_rng(seed)
    samples = np.sqrt(
        f_dist.rvs(nu1, nu2, size=n_mc, random_state=rng)
        * f_dist.rvs(nu1, nu2, size=n_mc, random_state=rng)
    )
    return float(np.mean(samples <= x))

# Replace p-value lines with:
cdf_val = _product_f_cdf(stat, nu1, nu2)
p_value = 2 * min(cdf_val, 1 - cdf_val)
```

### Verification test

Under H₀ with Σ₁ = Σ₂, generate data with p=2000, n₁=n₂=10, run 5000 reps, verify rejection rate at α=0.05 is in [0.035, 0.065].

---

## Bug 3 — Pooled covariance weighting in Schott (2007) variance estimator

### Severity: **LOW** (asymptotically negligible; finite-sample bias only)
### Affects
| Function | Lines (approx.) |
|---|---|
| `schott2007` | 997 |

### Paper reference
Schott, J. R. (2007). *A test for the equality of covariance matrices when the dimension is large relative to the sample sizes.* Comput. Statist. Data Anal., 51(12), 6535–6542.

### Description

The pooled covariance in the variance estimator τ̂ should be weighted by degrees of freedom (nᵢ = Nᵢ − 1), not raw sample sizes (Nᵢ).

**Current code (line ~997):**

```python
SsS = (Sxx * n1 + Syy * n2) / (n1 + n2)
```

Here `n1, n2` are `X.shape[0], Y.shape[0]` — the sample sizes N₁, N₂.

**Fixed code:**

```python
SsS = (Sxx * (n1 - 1) + Syy * (n2 - 1)) / (n1 + n2 - 2)
```

### Impact

For N₁=20, N₂=25, the relative error in tr(S²) is ~0.16%. Vanishes asymptotically.

### Verification test

Under H₀ (Σ = I), 20 000 reps with N₁=20, N₂=25, p=10. Verify type-I error rate at α=0.05 improves toward 0.05.

---

## Summary table

| # | Function(s) | Paper | Bug | Severity |
|---|---|---|---|---|
| 1 | `_srivastava_yanagihara_stat`, `_srivastava_2007_stat` | Srivastava & Yanagihara (2010), Thm 2.2 | Missing ×m factors in â₄ (3 of 5 terms) | HIGH |
| 2a | `ishii_two_sample` | Ishii (2017), §5 | `(h+1/h)/2` instead of `max(h, 1/h)` | HIGH |
| 2b | `ishii_two_sample` | Ishii (2017), Thm 5.1 | Simple F CDF instead of √(F₁·F₂) CDF | HIGH |
| 3 | `schott2007` | Schott (2007) | Pooled cov weighted by Nᵢ not nᵢ=Nᵢ−1 | LOW |
