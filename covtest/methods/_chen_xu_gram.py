"""
_chen_xu_gram.py
================

Shared building blocks for the Chen et al. (2010) and Xu et al. (2023) identity
covariance tests.

All quantities are derived from the centred Gram matrix G = Xc Xc' (n×n), where
Xc = X - mean(X).  Centering guarantees that R = 1'G1 = 0 and r = G1 = 0,
which dramatically simplifies the closed-form U-statistic expressions.

Summary of notation
-------------------
n     : sample size
p     : number of features
Xc    : n×p centred data matrix
G     : n×n Gram matrix  (G = Xc Xc')
d     : length-n vector of diagonal entries  dᵢ = ‖xᵢ − x̄‖²
D     : scalar  D  = Σᵢ dᵢ = tr(G)
D2    : scalar  D₂ = Σᵢ dᵢ² = Σᵢ ‖xᵢ − x̄‖⁴
Q2    : scalar  Q₂ = tr(G²) = ‖G‖_F²  (sum of squared entries of G)

Because Xc is centred, the U-statistic "row-sum" vector r = G·1 = 0.

Primary U-statistic building blocks (all estimators of tr(Σ) or tr(Σ²))
------------------------------------------------------------------------

Chen 2010 estimators (for identity test Vn):
  T1 = D / (n-1)                     unbiased estimator of tr(Σ)  [= tr(S)]
  Y2 = (Q2 - D2) / [n(n-1)]
  Y4 = (-Q2 + 2·D2) / [n(n-1)(n-2)]
  Y5 = (D² + 2·Q2 - 10·D2) / [n(n-1)(n-2)(n-3)]
  T2 = Y2 - 2·Y4 + Y5               unbiased estimator of tr(Σ²)   (needs n≥4)

Xu 2023 extra estimators (for adjusted test V̂):
  Ỹ2 = (D² - D2) / [n(n-1)]
  Ỹ4 = (-D² + 2·D2) / [n(n-1)(n-2)]
  T3 = Ỹ2 - 2·Ỹ4 + Y5              unbiased estimator of tr²(Σ)
  Y6 = D2 / n
  Y7 = -D2 / [n(n-1)]
  Y8 = Ỹ4                           (same object)
  δ̂  = Y6 - 4·Y7 + 2·Y8 + 4·Y4 - 3·Y5   5th-order U-statistic

References
----------
Chen, S. X., Zhang, L.-X., & Zhong, P.-S. (2010).
  "Tests for High-Dimensional Covariance Matrices."
  JASA, 105(490), 810-819.

Xu, G., et al. (2023 / 2025).
  "Adjusted location-invariant U-tests for the covariance matrix with
  elliptically high-dimensional data."
  Scandinavian Journal of Statistics, 52, 249-269.
"""

import numpy as np


def gram_blocks(X):
    """
    Compute all centred-Gram building blocks needed by Chen (2010) and Xu (2023).

    Parameters
    ----------
    X : array-like of shape (n, p)
        Raw data matrix. Centering is performed inside this function.

    Returns
    -------
    dict with keys:
        n, p, D, D2, Q2, Xc, G
    """
    X = np.asarray(X, dtype=np.float64)
    n, p = X.shape
    Xc = X - X.mean(axis=0)
    G = Xc @ Xc.T  # n×n Gram matrix
    d = np.einsum("ij,ij->i", Xc, Xc)  # squared norms (faster than np.diag(G))
    D = d.sum()  # tr(G)
    D2 = (d**2).sum()  # Σᵢ ‖yᵢ‖⁴
    Q2 = (G**2).sum()  # tr(G²) = ‖G‖_F²
    return dict(n=n, p=p, D=D, D2=D2, Q2=Q2, Xc=Xc, G=G, d=d)


# ─── Chen 2010 estimators ───────────────────────────────────────────────────


def T1_chen(blocks):
    """Unbiased estimator of tr(Σ): T₁ = D/(n-1) = tr(S)."""
    n, D = blocks["n"], blocks["D"]
    return D / (n - 1)


def T2_chen(blocks):
    """
    Unbiased estimator of tr(Σ²) via 4th-order U-statistics (Chen 2010).

    Requires n ≥ 4.

    T₂ = Y₂ − 2·Y₄ + Y₅

    where (for centred data, so R = 0 and r = 0):
      Y₂ = (Q₂ − D₂) / [n(n-1)]
      Y₄ = (−Q₂ + 2D₂) / [n(n-1)(n-2)]
      Y₅ = (D² + 2Q₂ − 10D₂) / [n(n-1)(n-2)(n-3)]
    """
    n, D, D2, Q2 = blocks["n"], blocks["D"], blocks["D2"], blocks["Q2"]
    if n < 4:
        raise ValueError(f"Chen 2010 T₂ requires n ≥ 4 (got n={n}).")
    Y2 = (Q2 - D2) / (n * (n - 1))
    Y4 = (-Q2 + 2 * D2) / (n * (n - 1) * (n - 2))
    Y5 = (D**2 + 2 * Q2 - 10 * D2) / (n * (n - 1) * (n - 2) * (n - 3))
    return Y2 - 2 * Y4 + Y5


# ─── Xu 2023 extra building blocks ──────────────────────────────────────────


def T3_xu(blocks):
    """
    Estimator of tr²(Σ) = [tr(Σ)]² (Xu 2023).

    T₃ = Ỹ₂ − 2·Ỹ₄ + Y₅

    where (for centred data):
      Ỹ₂ = (D² − D₂) / [n(n-1)]
      Ỹ₄ = (−D² + 2D₂) / [n(n-1)(n-2)]
      Y₅  = (D² + 2Q₂ − 10D₂) / [n(n-1)(n-2)(n-3)]

    Requires n ≥ 4.
    """
    n, D, D2, Q2 = blocks["n"], blocks["D"], blocks["D2"], blocks["Q2"]
    if n < 4:
        raise ValueError(f"Xu 2023 T₃ requires n ≥ 4 (got n={n}).")
    Y_tilde_2 = (D**2 - D2) / (n * (n - 1))
    Y_tilde_4 = (-(D**2) + 2 * D2) / (n * (n - 1) * (n - 2))
    Y5 = (D**2 + 2 * Q2 - 10 * D2) / (n * (n - 1) * (n - 2) * (n - 3))
    return Y_tilde_2 - 2 * Y_tilde_4 + Y5


def delta_hat_xu(blocks):
    """
    5th-order location-invariant U-statistic δ̂ₙ,ₚ (Xu 2023, Section 3).

    Estimates tr²(Σ)·E(R₁⁴)/E²(R₁²), which equals tr²(Σ) under Gaussian data.

    δ̂ = Y₆ − 4·Y₇ + 2·Y₈ + 4·Y₄ − 3·Y₅

    For centred data (r = 0):
      Y₆ = D₂ / n
      Y₇ = −D₂ / [n(n-1)]
      Y₈ = Ỹ₄ = (−D² + 2D₂) / [n(n-1)(n-2)]
      Y₄ = (−Q₂ + 2D₂) / [n(n-1)(n-2)]
      Y₅ = (D² + 2Q₂ − 10D₂) / [n(n-1)(n-2)(n-3)]

    Requires n ≥ 5.
    """
    n, D, D2, Q2 = blocks["n"], blocks["D"], blocks["D2"], blocks["Q2"]
    if n < 5:
        raise ValueError(f"Xu 2023 δ̂ requires n ≥ 5 (got n={n}).")
    Y6 = D2 / n
    Y7 = -D2 / (n * (n - 1))
    Y_tilde_4 = (-(D**2) + 2 * D2) / (n * (n - 1) * (n - 2))  # Y8 = Ỹ₄
    Y4 = (-Q2 + 2 * D2) / (n * (n - 1) * (n - 2))
    Y5 = (D**2 + 2 * Q2 - 10 * D2) / (n * (n - 1) * (n - 2) * (n - 3))
    return Y6 - 4 * Y7 + 2 * Y_tilde_4 + 4 * Y4 - 3 * Y5


def sigma2_hat_xu(delta, T3, n, p):
    """
    Estimated asymptotic variance σ̂²₀,ₙ,ₚ (Xu 2023, equation 14).

    σ̂²₀ = (2p²/n²)·{3p²/(p+2)²·(δ̂/T₃)² − 1}
          − (4p²/n²)·{p/(p+2)·(δ̂/T₃) − 1}

    Under Gaussian data, δ̂/T₃ → (p+2)/p, and σ̂²₀ → 4p²/n², so the
    adjusted test reduces to the Chen 2010 test statistic (n/2)·Vₙ.

    Parameters
    ----------
    delta : float  — δ̂ₙ,ₚ from `delta_hat_xu`
    T3    : float  — T₃ from `T3_xu`
    n, p  : int    — sample size and feature dimension

    Returns
    -------
    float  (always positive by construction; see Remark after eq. 14)
    """
    ratio = delta / T3  # estimates κ = E(R₁⁴)/E²(R₁²)
    factor = p**2 / n**2
    term1 = 2 * factor * (3 * p**2 / (p + 2) ** 2 * ratio**2 - 1)
    term2 = 4 * factor * (p / (p + 2) * ratio - 1)
    sigma2 = term1 - term2
    # Guard: σ̂² should be positive; use a small positive floor to be safe
    return max(sigma2, 1e-30)
