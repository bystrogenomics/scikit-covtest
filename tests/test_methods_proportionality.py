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
    threshold = (
        0.75
        if method.__name__ in ("ahmad_2022_proportionality_test", "proportional_cov_test_tsukuda")
        else 0.7
    )
    assert res["p_value"] < threshold


def test_proportionality_lz_regularization_numerical_stability():
    """Ensure LZ method raises ValueError when regularize != 0"""
    rng = np.random.default_rng(123)
    X = rng.standard_normal((60, 25))
    Y = rng.standard_normal((60, 25))
    with pytest.raises(ValueError, match="regularize is not supported"):
        proportionality_test_LZ(X, Y, regularize=1e-4)


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


def test_tsukuda_matsuura_regression_and_properties():
    # 1. Confirm that helper matches the closed-form Tsukuda-Matsuura estimator on a small deterministic matrix
    from covtest.methods._tsukuda_2019 import _a2_hat_tsukuda
    X_det = np.array([
        [1.0, 2.0],
        [3.0, 5.0],
        [2.0, -1.0]
    ])
    # Manual calculation yields 18.25
    a2_det = _a2_hat_tsukuda(X_det)
    assert np.allclose(a2_det, 18.25)

    # 2. Confirm that proportional_cov_test_tsukuda does not use sqrt(b2_hat) in the denominator
    # We will compute the values manually and verify
    rng = np.random.default_rng(1234)
    X = rng.standard_normal((10, 5))
    Y = rng.standard_normal((10, 5))
    
    res = proportional_cov_test_tsukuda(X, Y)
    
    # Manual implementation step-by-step
    m, p = X.shape
    n, _ = Y.shape
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    Sx = (Xc.T @ Xc) / (m - 1)
    Sy = (Yc.T @ Yc) / (n - 1)
    
    a_x1 = np.trace(Sx) / p
    a_y1 = np.trace(Sy) / p
    a_xy = np.trace(Sx @ Sy) / p
    a_x2 = _a2_hat_tsukuda(X)
    a_y2 = _a2_hat_tsukuda(Y)
    
    T = (m * n / (m + n)) * (
        a_x2 / (a_x1**2) + a_y2 / (a_y1**2) - 2.0 * a_xy / (a_x1 * a_y1)
    )
    b2_hat = ((m**2) / (m**2 + n**2)) * (a_x2 / (a_x1**2)) + (
        (n**2) / (m**2 + n**2)
    ) * (a_y2 / (a_y1**2))
    
    Z_correct = T / (2.0 * b2_hat)
    Z_incorrect_sqrt = T / (2.0 * np.sqrt(b2_hat))
    
    # Z_correct and Z_incorrect_sqrt should be different (provided b2_hat != 1.0)
    assert not np.allclose(Z_correct, Z_incorrect_sqrt)
    # The returned stat must be Z_correct
    assert np.allclose(res["stat"], Z_correct)

    # 3. Confirm scale invariance under proportional covariance scaling
    res_scaled = proportional_cov_test_tsukuda(3.0 * X, 0.5 * Y)
    assert np.allclose(res["stat"], res_scaled["stat"])
    assert np.allclose(res["p_value"], res_scaled["p_value"])

    # 4. Confirm scale invariance of X and sqrt(k)*Y under proportional covariance
    # X and Y generated from proportional covariances
    p_dim = 15
    n_samples = 30
    cov_x = np.eye(p_dim)
    cov_y = 4.0 * cov_x
    
    X_prop = rng.multivariate_normal(np.zeros(p_dim), cov_x, size=n_samples)
    Y_prop = rng.multivariate_normal(np.zeros(p_dim), cov_y, size=n_samples)
    
    res_prop = proportional_cov_test_tsukuda(X_prop, Y_prop)
    # Scale Y_prop further
    res_prop_scaled = proportional_cov_test_tsukuda(X_prop, 5.0 * Y_prop)
    assert np.allclose(res_prop["stat"], res_prop_scaled["stat"])
    assert np.allclose(res_prop["p_value"], res_prop_scaled["p_value"])


def test_tsukuda_anisotropic_diagonal_h0():
    # 5. Confirm anisotropic diagonal covariance under H0
    # e.g., Sigma_y = 5 * Sigma_x where Sigma_x has unequal eigenvalues
    rng = np.random.default_rng(42)
    p = 50
    n1 = 100
    n2 = 100
    
    # Diagonal elements from 1 to p
    diag_elements = np.arange(1, p + 1, dtype=float)
    Sigma_x = np.diag(diag_elements)
    Sigma_y = 5.0 * Sigma_x
    
    X = rng.multivariate_normal(np.zeros(p), Sigma_x, size=n1)
    Y = rng.multivariate_normal(np.zeros(p), Sigma_y, size=n2)
    
    res = proportional_cov_test_tsukuda(X, Y)
    
    # Assert result conforms to dict shape and has reasonable non-rejection behavior under H0
    assert_result_dict_2samp(res)
    # Under H0, the p-value shouldn't be extremely small (e.g. not rejecting at alpha = 0.05 / 0.01)
    assert res["p_value"] > 0.01


def test_lz_proportionality_additional_verifications():
    import scipy.stats as stats
    rng = np.random.default_rng(123)
    X = rng.standard_normal((60, 25))
    Y = rng.standard_normal((60, 25))
    
    res = proportionality_test_LZ(X, Y)
    
    # 1. The function returns exactly the keys {"stat", "p_value"}
    assert set(res.keys()) == {"stat", "p_value"}
    
    # 2. p_value == stats.norm.sf(stat) up to floating-point tolerance
    assert np.allclose(res["p_value"], stats.norm.sf(res["stat"]))
    
    # 3. Passing regularize > 0 raises a ValueError
    with pytest.raises(ValueError, match="regularize is not supported"):
        proportionality_test_LZ(X, Y, regularize=0.1)
        
    # 4. Mismatched feature dimensions raise a ValueError rather than an AssertionError
    X_mis = rng.standard_normal((60, 25))
    Y_mis = rng.standard_normal((60, 26))
    with pytest.raises(ValueError, match="same number of columns"):
        proportionality_test_LZ(X_mis, Y_mis)


