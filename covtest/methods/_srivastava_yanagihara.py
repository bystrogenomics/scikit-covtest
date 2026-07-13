"""
srivastava_yanagihara.py
========================

Estimators for the Srivastava, Yanagihara & Kubokawa (2014) covariance tests.

Implements the unbiased O(N²) estimator of a₂ = tr(Σ²)/p that is valid under
non-normal distributions, as defined in equation (2.5) of the paper.

References
----------
Srivastava, M. S., Yanagihara, H., & Kubokawa, T. (2014).
"Tests for covariance matrices in high dimension with less sample size."
Journal of Multivariate Analysis, 130, 289-309.
https://doi.org/10.1016/j.jmva.2014.06.003
"""

import numpy as np


def a_1_hat(Y):
    """
    Estimate a₁ = tr(Σ)/p from data matrix Y.

    Uses the unbiased estimator â₁ = tr(S)/p (equation 2.3),
    where S = V/n is the sample covariance and V = Yc'Yc is the scatter matrix.

    Parameters
    ----------
    Y : array-like of shape (N, p)
        Raw data matrix (centering is performed internally).

    Returns
    -------
    float
        Unbiased estimate of a₁ = tr(Σ)/p.
    """
    Yc = Y - Y.mean(axis=0)
    N, p = Yc.shape
    n = N - 1  # degrees of freedom
    # tr(V) = tr(n*S) = n * tr(S)/p * p = n * a_1 * p  =>  a_1 = tr(V)/(n*p)
    tr_V = np.sum(Yc ** 2)  # tr(Yc.T @ Yc) = sum of all squared entries
    return tr_V / (n * p)


def a_2_hat(Y):
    """
    Estimate a₂ = tr(Σ²)/p via the new unbiased O(N²) estimator.

    Implements equation (2.5) of Srivastava et al. (2014), which is unbiased
    under a general class of distributions (no normality required):

        â₂ = [(N-2)·n·tr(M²) - N·n·tr(D²) + (tr(D))²] / f

    where n = N-1, M = Yc Yc' (the N×N Gram matrix of centered observations),
    D is the N×N diagonal matrix with Dᵢᵢ = ‖yᵢ‖², and f = p·N·(N-1)·(N-2)·(N-3).

    Note: tr(M²) = tr(V²) where V = Yc' Yc (p×p scatter matrix), computed
    as the Frobenius-norm-squared of the smaller of M or V.

    Parameters
    ----------
    Y : array-like of shape (N, p)
        Raw data matrix (centering is performed internally).

    Returns
    -------
    float
        Unbiased estimate of a₂ = tr(Σ²)/p.
    """
    Y = np.asarray(Y, dtype=np.float64)
    N, p = Y.shape
    n = N - 1  # paper's n = N - 1

    # Center the data: yᵢ = xᵢ - x̄
    Yc = Y - Y.mean(axis=0)

    # tr(M²) = tr(G²) = tr(V²) where G = Yc Yc' (N×N) and V = Yc' Yc (p×p).
    # Use the smaller of the two for efficiency.
    if N <= p:
        G = Yc @ Yc.T  # N×N Gram matrix
        tr_M2 = (G ** 2).sum()  # = tr(G²) = tr(M²)
    else:
        V = Yc.T @ Yc  # p×p scatter matrix
        tr_M2 = (V ** 2).sum()  # = tr(V²) = tr(M²)

    norms_sq = np.sum(Yc ** 2, axis=1)  # ‖yᵢ‖² for each observation
    tr_D2 = np.sum(norms_sq ** 2)  # tr(D²) = Σᵢ ‖yᵢ‖⁴
    tr_D = np.sum(norms_sq)  # tr(D) = Σᵢ ‖yᵢ‖² = tr(V)

    f = p * N * (N - 1) * (N - 2) * (N - 3)
    numerator = (N - 2) * n * tr_M2 - N * n * tr_D2 + tr_D ** 2

    return numerator / f


def gamma_2_hat(Y):
    """
    Estimate γ₂ = a₂ - 2a₁ (the unstandardised identity distance measure).

    Under H₀: Σ = Iₚ, γ₂ = a₂ - 2a₁ = 1 - 2 = -1, so γ₂ + 1 = 0.

    Parameters
    ----------
    Y : array-like of shape (N, p)

    Returns
    -------
    float
    """
    return a_2_hat(Y) - 2.0 * a_1_hat(Y)
