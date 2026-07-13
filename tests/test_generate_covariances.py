# test_generate_covariances.py
import numpy as np
import pytest

import covtest.simulation.generate_covariances as gc


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def is_symmetric(mat, tol=1e-8):
    return np.allclose(mat, mat.T, atol=tol)


def is_pos_def(mat):
    return np.all(np.linalg.eigvalsh(mat) > 0)


def test_sample_goe(rng):
    A = gc.sample_goe(5, rng)
    assert A.shape == (5, 5)
    assert is_symmetric(A)


def test_sample_cov_uniform_correlation_valid_levels(rng):
    for level in ["low", "medium", "high"]:
        R = gc.sample_cov_uniform_correlation(4, level=level, rng=rng)
        assert R.shape == (4, 4)
        assert is_symmetric(R)
        assert is_pos_def(R)


def test_sample_cov_uniform_correlation_invalid_level(rng):
    with pytest.raises(ValueError):
        gc.sample_cov_uniform_correlation(3, level="invalid", rng=rng)


def test_generate_spectral_cov(rng):
    cov = gc.generate_spectral_cov(6, rng)
    assert cov.shape == (6, 6)
    assert is_symmetric(cov)
    assert is_pos_def(cov)


def test_generate_toeplitz_cov():
    cov = gc.generate_toeplitz_cov(4, 0.5)
    assert cov.shape == (4, 4)
    assert is_symmetric(cov)
    assert np.allclose(np.diag(cov), 1.0)


def test_generate_block_diagonal_cov(rng):
    cov = gc.generate_block_diagonal_cov(6, block_size=2, rng=rng)
    assert cov.shape == (6, 6)
    assert is_symmetric(cov)


def test_generate_low_rank_cov(rng):
    cov = gc.generate_low_rank_cov(5, rank=2, noise_var=0.1, rng=rng)
    assert cov.shape == (5, 5)
    assert is_symmetric(cov)
    assert is_pos_def(cov)


def test_generate_sparse_precision_cov(rng):
    cov = gc.generate_sparse_precision_cov(5, sparsity=0.5, rng=rng)
    assert cov.shape == (5, 5)
    assert is_symmetric(cov)
    assert is_pos_def(cov)


def test_generate_marchenko_pastur(rng):
    cov = gc.generate_marchenko_pastur(5, 20, rng)
    assert cov.shape == (5, 5)
    assert is_symmetric(cov)
    assert is_pos_def(cov)


def test_generate_spiked_covariance(rng):
    cov = gc.generate_spiked_covariance(5, spike_eigenvalue=10.0, num_spikes=2, rng=rng)
    assert cov.shape == (5, 5)
    assert is_symmetric(cov)
    assert is_pos_def(cov)


def test_generate_spiked_covariance_invalid_num_spikes(rng):
    with pytest.raises(ValueError):
        gc.generate_spiked_covariance(3, num_spikes=5, rng=rng)
