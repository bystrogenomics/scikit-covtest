import numpy as np
import pytest

from covtest.methods.hypothesis_spherical import (
    _john_stat,
    _spatial_sign_rows,
    _U_tensor,
    bartlett_sphericity_test,
    czz_sphericity_test,
    hallin_rank_sphericity_test,
    john_sphericity,
    sk_test,
    srivastava_2005_sphericity,
)


@pytest.fixture
def identity_data():
    rng = np.random.default_rng(0)
    return rng.normal(size=(30, 4))


@pytest.fixture
def non_spherical_data():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(80, 4))
    return X @ np.diag([1, 2, 3, 4])  # breaks sphericity


def test_john_stat_float(identity_data):
    stat = _john_stat(identity_data)
    assert isinstance(stat, float)


def test_john_sphericity_output(identity_data):
    res = john_sphericity(identity_data)
    assert set(res.keys()) == {"stat", "p_value"}
    assert isinstance(res["stat"], float)
    assert 0 <= res["p_value"] <= 1


def test_czz_output(identity_data):
    res = czz_sphericity_test(identity_data)
    assert "stat" in res and "p_value" in res
    assert isinstance(res["stat"], float)
    assert 0 <= res["p_value"] <= 1


def test_bartlet_output(identity_data):
    res = bartlett_sphericity_test(identity_data)
    assert "stat" in res and "p_value" in res
    assert isinstance(res["stat"], float)
    assert 0 <= res["p_value"] <= 1


def test_hallin_output(identity_data):
    res = hallin_rank_sphericity_test(identity_data)
    assert "stat" in res and "p_value" in res
    assert isinstance(res["stat"], float)
    assert 0 <= res["p_value"] <= 1

    res = hallin_rank_sphericity_test(identity_data, method="vdw")
    assert "stat" in res and "p_value" in res
    assert isinstance(res["stat"], float)
    assert 0 <= res["p_value"] <= 1


def test_srivastava_output(identity_data):
    res = srivastava_2005_sphericity(identity_data)
    assert "stat" in res and "p_value" in res
    assert isinstance(res["stat"], float)
    assert 0 <= res["p_value"] <= 1


def test_spatial_sign_rows_and_tensor(identity_data):
    A = identity_data[:5, :]  # small slice
    signs = _spatial_sign_rows(A)
    assert np.allclose(np.linalg.norm(signs, axis=1), 1.0, atol=1e-8)

    U = _U_tensor(A)
    n, _, p = U.shape
    assert U.shape == (A.shape[0], A.shape[0], A.shape[1])
    assert np.allclose(U[np.arange(n), np.arange(n)], 0.0)


def test_sk_test_output(identity_data):
    res = sk_test(identity_data)
    assert set(res.keys()) == {"stat", "p_value"}
    assert isinstance(res["stat"], float)
    assert 0 <= res["p_value"] <= 1


def test_sk_test_rejects_small_n():
    X = np.random.normal(size=(3, 2))
    with pytest.raises(ValueError):
        sk_test(X)


def test_non_spherical_data_gives_signal(non_spherical_data):
    john_res = john_sphericity(non_spherical_data)
    sk_res = sk_test(non_spherical_data)
    hl_res1 = hallin_rank_sphericity_test(non_spherical_data, method="wilcoxon")
    hl_res2 = hallin_rank_sphericity_test(non_spherical_data, method="vdw")
    sk_res = sk_test(non_spherical_data)
    czz_res = czz_sphericity_test(non_spherical_data)
    # Expect evidence against null, so p-value should not be 1
    thresh = 0.1
    assert john_res["p_value"] < thresh
    assert sk_res["p_value"] < thresh
    assert hl_res1["p_value"] < thresh
    assert hl_res2["p_value"] < thresh
    assert czz_res["p_value"] < thresh
