"""
Data Generation Utilities
=========================

This module provides functions for generating multivariate data with a prescribed
covariance structure under a variety of heavy-tailed and non-Gaussian distributions.
It is designed for simulation studies, stress-testing covariance-based hypothesis
tests, and exploring robustness of statistical procedures in high-dimensional
settings.

Implemented functionality
-------------------------
- `generate_heavy_tailed_samples`: Generate synthetic data with exact covariance
  but marginal distributions drawn from various heavy-tailed or non-Gaussian laws:

  * Student's t-distribution (requires degrees of freedom `df`).
  * Laplace distribution (double exponential).
  * Log-normal distribution (with shape parameter `s`).
  * Symmetric Pareto distribution (with shape parameter `b`).
  * Gaussian scale mixtures (gamma-distributed scaling factors).
  * Variance-gamma mixtures.
  * Generalized hyperbolic–like approximations.

Each distribution is normalized to enforce zero mean and unit variance before
covariance shaping. The covariance structure is imposed via Cholesky
factorization of the target covariance matrix.

Parameters
----------
- `cov` : array_like of shape (p, p)
  Target positive-definite covariance matrix.
- `n` : int
  Number of samples to generate.
- `dist_type` : str
  Choice of distribution family, one of {'t','laplace','lognormal','pareto',
  'scale_mixture','variance_gamma','gh'}.
- `rng` : numpy.random.Generator, optional
  Random number generator for reproducibility.
- `options` : dict, optional
  Distribution-specific parameters (e.g., `df`, `s`, `b`, `a`).

Returns
-------
- `np.ndarray` : Array of shape (n, p) containing the generated multivariate samples.

Notes
-----
The generated samples always respect the prescribed covariance structure
by applying the transformation:

$$ X = Z L^T, $$

where $Z$ are standardized heavy-tailed draws and $L$ is the Cholesky factor
of the covariance matrix.
"""

from typing import Any, Dict, Optional

import numpy as np
from numpy.random import default_rng
from scipy.linalg import cholesky as chol_scipy
from scipy.stats import gamma, laplace, lognorm, pareto, t


def generate_heavy_tailed_samples(cov, n, dist_type, rng=None, options=None):
    """
    Generate N heavy-tailed multivariate samples with exact covariance structure.

    Parameters:
        cov (np.ndarray): (p x p) desired positive-definite covariance matrix.
        n (int): Number of samples to generate.
        dist_type (str): One of the following options:
            - 't': Student's t-distribution (requires 'df' in options)
            - 'laplace': Laplace distribution (no options required)
            - 'lognormal': Log-normal distribution (options: {'s': shape parameter})
            - 'pareto': Symmetric Pareto (options: {'b': shape parameter})
            - 'scale_mixture': Gaussian scale mixture (options: {'a': shape for gamma})
            - 'variance_gamma': Gamma-normal mixture (options: {'a': shape})
            - 'gh': Generalized hyperbolic approx (options: {'a': shape})

        rng (np.random.Generator): Optional numpy random generator.
        options (dict): Optional dictionary of distribution parameters.

    Returns:
        np.ndarray: (n x p) multivariate samples.
    """
    if rng is None:
        rng = default_rng()
    if options is None:
        options = {}

    p = cov.shape[0]
    chol = chol_scipy(cov)  # default upper-triangular

    if dist_type == "normal":
        Z = rng.normal(size=(n, p))
    elif dist_type == "t":
        df = options.get("df", 3)
        Z = t(df=df).rvs(size=(n, p), random_state=rng)
        Z /= np.sqrt(df / (df - 2))
    elif dist_type == "laplace":
        Z = laplace.rvs(size=(n, p), random_state=rng)
        Z /= np.sqrt(2)
    elif dist_type == "lognormal":
        s = options.get("s", 1.0)
        Z = lognorm(s=s).rvs(size=(n, p), random_state=rng)
        Z = (Z - Z.mean(axis=0)) / Z.std(axis=0)
    elif dist_type == "pareto":
        b = options.get("b", 3.0)
        X = (
            pareto(b=b).rvs(size=(n, p), random_state=rng) - 1.0
        )  # shift support to [0, ∞)
        signs = rng.choice([-1, 1], size=(n, p))
        Z = X * signs
        Z = (Z - Z.mean(axis=0)) / Z.std(axis=0)
    elif dist_type == "scale_mixture":
        a = options.get("a", 2.0)
        scales = 1 / np.sqrt(gamma(a=a, scale=1).rvs(size=n, random_state=rng))
        Z = rng.normal(size=(n, p)) * scales[:, None]
    elif dist_type == "variance_gamma":
        a = options.get("a", 1.0)
        gammas = gamma(a=a, scale=1.0).rvs(size=n, random_state=rng)
        Z = rng.normal(size=(n, p)) * np.sqrt(gammas[:, None])
        Z = (Z - Z.mean(axis=0)) / Z.std(axis=0)
    elif dist_type == "gh":
        a = options.get("a", 1.0)
        taus = 1 / gamma(a=a, scale=1.0).rvs(size=n, random_state=rng)
        Z = rng.normal(size=(n, p)) * np.sqrt(taus[:, None])
        Z = (Z - Z.mean(axis=0)) / Z.std(axis=0)
    else:
        raise ValueError(f"Unsupported distribution type: {dist_type}")

    return Z @ chol


def generate_heavy_tailed_alternative(
    cov: np.ndarray,
    n: int,
    dist_type: str,
    rng: Optional[np.random.Generator] = None,
    options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Generate heavy-tailed samples under several alternatives, using
    generate_heavy_tailed_samples for the noise component.

    Parameters
    ----------
    cov : array, shape (p, p)
        Baseline covariance matrix for the null model.
    n : int
        Number of samples to generate.
    dist_type : str
        Distribution type for the heavy-tailed noise. Passed to
        generate_heavy_tailed_samples.
    rng : numpy.random.Generator, optional
        Random number generator. If None, uses default_rng().
    options : dict, optional
        Configuration for both the base noise and the alternative.
        To avoid collisions with distribution options, you may place
        distribution parameters under options['dist'].

        Common keys:
        - 'alt': one of {'mixture', 'scaled_cov', 'rank1_bump', 'eig_bump'}.
                  Default is 'mixture'.

        Mixture alternative (alt == 'mixture'):
        - 'p_mix': float in [0,1], probability of the +mu component.
                   Default 0.5. With 0.5 the mean is zero.
        - 'mu': array shape (p,), optional. If provided, used directly.
        - 'mu_norm': float > 0, target norm of mu if 'mu' not given.
                     Default 1.0.
        - 'mu_dir': array shape (p,), optional direction to define mu.
                    If not given, a random unit vector is used.

        Scaled covariance (alt == 'scaled_cov'):
        - 'scale_k': float > 0, scalar multiplier for the covariance.
                     Default 2.0.

        Rank-1 bump (alt == 'rank1_bump'):
        - 'v': array shape (p,), optional. If provided, use cov + v v^T.
        - 'v_norm': float > 0, if 'v' not given, a random unit vector
                    scaled to this norm is used. Default 1.0.
        - 'v_dir': array shape (p,), optional direction to define v.

        Top eigen bump (alt == 'eig_bump'):
        - 'eig_k': float >= 0, amount to add to the largest eigenvalue.
                   Default 1.0.

        Distribution options:
        - Put parameters for the base noise under options['dist'].
          For example: options={'dist': {'df': 5}} for dist_type='t'.

    Returns
    -------
    result : dict
        Keys:
        - 'X'        : array shape (n, p), generated samples.
        - 'alt'      : str, which alternative was used.
        - 'cov_used' : array shape (p, p), covariance passed to the
                       heavy-tailed sampler for this draw.
        - 'details'  : dict of alternative-specific details, such as
                       'mu', 'labels' for mixture, 'k', 'v', or 'u_max'.

    Notes
    -----
    - Mixture with p_mix = 0.5 yields mean zero and increases the
      covariance by mu mu^T in expectation, on top of the base noise
      covariance. This provides a symmetric two-component alternative.
    - 'rank1_bump' produces cov_new = cov + v v^T, which is a spiked
      covariance along direction v.
    - 'eig_bump' produces cov_new = cov + eig_k * u_max u_max^T, where
      u_max is the top eigenvector of cov. This targets the leading mode.
    - All cases inherit heavy-tailed behavior from dist_type via
      generate_heavy_tailed_samples.
    """
    if rng is None:
        rng = np.random.default_rng()

    if options is None:
        options = {}

    alt = options.get("alt", "mixture")
    p = cov.shape[0]
    cov = np.asarray(cov, dtype=float)
    cov = 0.5 * (cov + cov.T)  # symmetrize for numerical robustness

    # Split distribution options to avoid collisions with alt keys
    dist_opts = options.get("dist", None)

    def _unit_vector_from_dir_or_random(
        dir_vec: Optional[np.ndarray],
    ) -> np.ndarray:
        if dir_vec is None:
            v = rng.standard_normal(p)
        else:
            v = np.asarray(dir_vec, dtype=float)
        norm = np.linalg.norm(v)
        if norm == 0:
            raise ValueError("Direction vector has zero norm.")
        return v / norm

    if alt == "mixture":
        # Base heavy-tailed noise under the null covariance
        X0 = generate_heavy_tailed_samples(
            cov, n, dist_type, rng=rng, options=dist_opts
        )

        # Build mu
        mu = options.get("mu", None)
        if mu is None:
            mu_norm = float(options.get("mu_norm", 1.0))
            mu_dir = options.get("mu_dir", None)
            u = _unit_vector_from_dir_or_random(mu_dir)
            mu = mu_norm * u
        else:
            mu = np.asarray(mu, dtype=float)
            if mu.shape != (p,):
                raise ValueError("mu must have shape (p,)")

        p_mix = float(options.get("p_mix", 0.5))
        if not (0.0 <= p_mix <= 1.0):
            raise ValueError("p_mix must be in [0, 1]")

        # Labels s in {+1, -1}, P(+1) = p_mix
        s = rng.random(n) < p_mix
        s = np.where(s, 1.0, -1.0)  # shape (n,)

        X = X0 + s[:, None] * mu[None, :]

        return {
            "X": X,
            "alt": "mixture",
            "cov_used": cov,  # base covariance used for the noise component
            "details": {
                "mu": mu,
                "p_mix": p_mix,
                "labels": s.astype(int),  # +1 or -1 per sample
            },
        }

    elif alt == "scaled_cov":
        k = float(options.get("scale_k", 2.0))
        if k <= 0.0:
            raise ValueError("scale_k must be positive.")
        cov_new = k * cov
        X = generate_heavy_tailed_samples(
            cov_new, n, dist_type, rng=rng, options=dist_opts
        )
        return {
            "X": X,
            "alt": "scaled_cov",
            "cov_used": cov_new,
            "details": {"k": k},
        }

    elif alt == "rank1_bump":
        v = options.get("v", None)
        if v is None:
            v_dir = options.get("v_dir", None)
            v_norm = float(options.get("v_norm", 1.0))
            u = _unit_vector_from_dir_or_random(v_dir)
            v = v_norm * u
        else:
            v = np.asarray(v, dtype=float)
            if v.shape != (p,):
                raise ValueError("v must have shape (p,)")

        cov_new = cov + np.outer(v, v)
        X = generate_heavy_tailed_samples(
            cov_new, n, dist_type, rng=rng, options=dist_opts
        )
        return {
            "X": X,
            "alt": "rank1_bump",
            "cov_used": cov_new,
            "details": {"v": v},
        }

    elif alt == "eig_bump":
        eig_k = float(options.get("eig_k", 1.0))
        if eig_k < 0.0:
            raise ValueError("eig_k must be nonnegative.")
        # Top eigenvector
        vals, vecs = np.linalg.eigh(cov)
        idx = int(np.argmax(vals))
        u_max = vecs[:, idx]
        cov_new = cov + eig_k * np.outer(u_max, u_max)
        X = generate_heavy_tailed_samples(
            cov_new, n, dist_type, rng=rng, options=dist_opts
        )
        return {
            "X": X,
            "alt": "eig_bump",
            "cov_used": cov_new,
            "details": {"eig_k": eig_k, "u_max": u_max},
        }

    else:
        raise ValueError(
            "Unknown alt. Expected one of {'mixture','scaled_cov','rank1_bump','eig_bump'}."
        )
