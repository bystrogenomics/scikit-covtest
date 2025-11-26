import numpy as np
import pytest
from numpy.linalg import eigh
from numpy.random import default_rng

from covtest.simulation.generate_data import generate_heavy_tailed_samples


@pytest.mark.parametrize(
    "dist_type,options",
    [
        ("t", {"df": 4}),
        ("laplace", None),
        ("lognormal", {"s": 0.8}),
        ("pareto", {"b": 3.5}),
        ("scale_mixture", {"a": 2.0}),
        ("variance_gamma", {"a": 1.0}),
        ("gh", {"a": 1.5}),
    ],
)
def test_heavy_tailed_distribution_shapes_and_covariance(dist_type, options):
    rng = default_rng(123)
    n, p = 5000, 4
    cov = np.diag(np.linspace(1, 2, p))

    samples = generate_heavy_tailed_samples(cov, n, dist_type, rng, options)

    # Test shape
    assert samples.shape == (n, p), f"Incorrect shape for {dist_type}"

    # Test empirical covariance matches target (up to scaling for some dists)
    emp_cov = np.cov(samples, rowvar=False)
    eigvals = eigh(emp_cov @ np.linalg.inv(cov))[0]

    # Accept if eigenvalues are close to 1 (allowing tolerance)
    assert np.all(
        (eigvals > 0.5) & (eigvals < 2.0)
    ), f"Empirical covariance deviates too much for {dist_type}"


def test_invalid_distribution_raises():
    cov = np.eye(3)
    with pytest.raises(ValueError):
        generate_heavy_tailed_samples(cov, 100, "unknown_dist")


def test_default_rng_and_options():
    cov = np.eye(2)
    samples = generate_heavy_tailed_samples(cov, 100, "t", options={"df": 4})
    assert samples.shape == (100, 2)


def test_covariance_output_is_symmetric():
    cov = np.diag([1, 2, 3])
    samples = generate_heavy_tailed_samples(cov, 1000, "t", options={"df": 4})
    emp_cov = np.cov(samples, rowvar=False)
    assert np.allclose(emp_cov, emp_cov.T, atol=1e-6)


# NOTE: To run these tests, save to a file `test_heavy_tailed.py` and run:
# pytest test_heavy_tailed.py
"Tests generated successfully."
