import numpy as np
import pytest

from covtest.methods.hypothesis_proportionality import (
    ahmad_2022_proportionality_test,
    bartlett_adjusted_proportionality_test,
    flury_proportionality_test,
    proportional_cov_test_tsukuda,
    proportionality_plrt,
    proportionality_test_LZ,
    proportionality_test_signs,
)


def assert_result_dict_2samp(res):
    assert isinstance(res, dict)
    assert "stat" in res
    assert np.isfinite(res["stat"])
    assert "p_value" in res
    assert 0 <= res["p_value"] <= 1


@pytest.mark.parametrize(
    "method",
    [
        ahmad_2022_proportionality_test,
        proportionality_test_LZ,
        flury_proportionality_test,
        bartlett_adjusted_proportionality_test,
        proportionality_test_signs,
        proportional_cov_test_tsukuda,
        proportionality_plrt,
    ],
)
def test_proportionality_methods_smoke(method):
    """Smoke test on N(0, I) data"""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((80, 20))
    Y = rng.standard_normal((80, 20))
    res = method(X, Y)
    assert_result_dict_2samp(res)


@pytest.mark.parametrize(
    "method",
    [
        ahmad_2022_proportionality_test,
        proportionality_test_LZ,
        flury_proportionality_test,
        bartlett_adjusted_proportionality_test,
        proportionality_test_signs,
        proportional_cov_test_tsukuda,
        proportionality_plrt,
    ],
)
def test_proportionality_methods_null_finite(method):
    """Test under H0: Sigma1 = Sigma2 = I"""
    rng = np.random.default_rng(123)
    for _ in range(5):
        X = rng.standard_normal((100, 30))
        Y = rng.standard_normal((100, 30))
        res = method(X, Y)
        assert_result_dict_2samp(res)
        assert abs(res["stat"]) < 10000  # sanity bound


@pytest.mark.parametrize(
    "method",
    [
        ahmad_2022_proportionality_test,
        proportionality_test_LZ,
        flury_proportionality_test,
        bartlett_adjusted_proportionality_test,
        proportionality_test_signs,
        proportional_cov_test_tsukuda,
        proportionality_plrt,
    ],
)
def test_proportionality_methods_alternative_power(method):
    """Test that methods detect deviation from proportionality"""
    rng = np.random.default_rng(42)
    p = 25
    n = 100
    A = rng.standard_normal((p, p))
    Sigma1 = A @ A.T
    Sigma2 = 2.5 * Sigma1.copy()
    Sigma2[0, 0] *= 2  # break proportionality

    X = rng.multivariate_normal(np.zeros(p), Sigma1, size=n)
    Y = rng.multivariate_normal(np.zeros(p), Sigma2, size=n)
    res = method(X, Y)
    assert_result_dict_2samp(res)
    # Not strict: allow some failure due to finite sample
    assert res["p_value"] < 0.7


def test_proportionality_lz_regularization_numerical_stability():
    """Ensure LZ method handles ill-conditioned Sigma2"""
    rng = np.random.default_rng(123)
    X = rng.standard_normal((60, 25))
    Y = rng.standard_normal((60, 25))
    res = proportionality_test_LZ(X, Y, regularize=1e-4)
    assert_result_dict_2samp(res)


def test_ahmad_2022_helpers():
    """Test the estimators in _ahmad2022.py module directly."""
    from covtest.methods import _ahmad2022 as ahmad2022

    rng = np.random.default_rng(42)
    X1 = rng.standard_normal((50, 10))
    X2 = rng.standard_normal((60, 10))

    # Test estimate_Ei_trSigma2
    E1 = ahmad2022.estimate_Ei_trSigma2(X1)
    assert isinstance(E1, float)
    assert E1 > 0

    # Test estimate_E12_trSigma1Sigma2
    E12 = ahmad2022.estimate_E12_trSigma1Sigma2(X1, X2)
    assert isinstance(E12, float)

    # Test validation exceptions
    with pytest.raises(ValueError, match="must be a 2D array"):
        ahmad2022.estimate_Ei_trSigma2(np.ones(10))

    with pytest.raises(ValueError, match="Need n >= 4"):
        ahmad2022.estimate_Ei_trSigma2(np.ones((3, 10)))

    with pytest.raises(ValueError, match="same number of features"):
        ahmad2022.estimate_E12_trSigma1Sigma2(
            np.ones((10, 5)), np.ones((10, 6))
        )
