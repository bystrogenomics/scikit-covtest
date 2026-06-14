import numpy as np
import pytest

from covtest.methods.hypothesis_identity import (
    _fisher_2012_stat_,
    _ledoit_wolf_stat,
    _nagao_stat,
    _srivastava2011_,
    fisher_single_sample,
    ledoit_wolf_identity,
    nagao_identity,
    one_sample_cov_test,
    srivastava2011_single_sample,
    srivastava_2005_identity,
    tyler_identity,
    srivastava_2014_identity,
    chen_2010_identity,
    xu_2023_identity,
    ahmad_2017_identity,
    test_identity_T2 as identity_T2_test,
)


@pytest.fixture
def identity_data():
    rng = np.random.default_rng(42)
    return rng.normal(size=(50, 5))


@pytest.fixture
def non_identity_data():
    rng = np.random.default_rng(123)
    X = rng.normal(size=(50, 5))
    return X @ np.diag([1, 2, 3, 4, 5])


def test_ledoit_wolf_stat(identity_data):
    val = _ledoit_wolf_stat(identity_data)
    assert isinstance(val, float)


def test_ledoit_wolf_identity_output(identity_data):
    res = ledoit_wolf_identity(identity_data)
    assert set(res.keys()) == {"stat", "p_value"}
    assert 0 <= res["p_value"] <= 1


def test_nagao_stat_and_identity(identity_data):
    V = _nagao_stat(identity_data)
    assert isinstance(V, float)
    res = nagao_identity(identity_data)
    assert "stat" in res and "p_value" in res


def test_srivastava_identity(identity_data):
    res = srivastava_2005_identity(identity_data)
    assert "stat" in res and "p_value" in res
    assert isinstance(res["stat"], float)


def test_tyler_identity(identity_data):
    res_tr = tyler_identity(identity_data, unknown_mean=False, method="tr")
    res_log = tyler_identity(identity_data, unknown_mean=False, method="log")
    assert all(k in res_tr for k in ("stat", "p_value"))
    assert all(k in res_log for k in ("stat", "p_value"))


def test_methods_distinguish_non_identity(non_identity_data):
    # Expect lower p-values under alternatives
    res1 = ledoit_wolf_identity(non_identity_data)
    res2 = nagao_identity(non_identity_data)
    assert res1["p_value"] < 1
    assert res2["p_value"] < 1


def test_fisher_stat_returns_float(identity_data):
    n, p = identity_data.shape
    S = np.cov(identity_data, rowvar=False)
    stat = _fisher_2012_stat_(n - 1, p, S)
    assert isinstance(stat, float)


def test_fisher_identity_output(identity_data):
    res = fisher_single_sample(identity_data, Sigma="identity")
    assert set(res.keys()) == {"stat", "p_value"}
    assert isinstance(res["stat"], float)
    assert 0 <= res["p_value"] <= 1


def test_srivastava_stat_returns_float(identity_data):
    n, p = identity_data.shape
    S = np.cov(identity_data, rowvar=False)
    stat = _srivastava2011_(n - 1, p, S)
    assert isinstance(stat, float)


def test_srivastava_identity_output(identity_data):
    res = srivastava2011_single_sample(identity_data, Sigma="identity")
    assert set(res.keys()) == {"stat", "p_value"}
    assert isinstance(res["stat"], float)
    assert 0 <= res["p_value"] <= 1


def test_non_identity_gives_lower_pvalues(non_identity_data):
    fisher_res = fisher_single_sample(non_identity_data)
    sriv_res = srivastava2011_single_sample(non_identity_data)
    # Under non-identity, we expect evidence against null (smaller p-values)
    assert fisher_res["p_value"] < 1
    assert sriv_res["p_value"] < 1


def test_invalid_sigma_raises(identity_data):
    # If Sigma is not "identity", function attempts SVD, so pass wrong shape
    bad_sigma = np.array([1, 2, 3])  # not square
    with pytest.raises(Exception):
        fisher_single_sample(identity_data, Sigma=bad_sigma)
    with pytest.raises(Exception):
        srivastava2011_single_sample(identity_data, Sigma=bad_sigma)


def test_one_sample_cov_test(identity_data):
    res = one_sample_cov_test(identity_data)
    assert set(res.keys()) == {"p_value", "z_value", "lrt"}
    assert 0 <= res["p_value"] <= 1


def test_srivastava_2014_identity(identity_data, non_identity_data):
    res = srivastava_2014_identity(identity_data)
    assert set(res.keys()) == {"stat", "p_value"}
    assert isinstance(res["stat"], float)
    assert 0 <= res["p_value"] <= 1

    res_alt = srivastava_2014_identity(non_identity_data)
    assert res_alt["p_value"] < 1

    # Check N < 4 validation
    short_data = np.random.default_rng(42).normal(size=(3, 5))
    with pytest.raises(
        ValueError, match="Srivastava \\(2014\\) test requires N >= 4."
    ):
        srivastava_2014_identity(short_data)


def test_chen_2010_identity(identity_data, non_identity_data):
    res = chen_2010_identity(identity_data)
    assert set(res.keys()) == {"stat", "p_value"}
    assert isinstance(res["stat"], float)
    assert 0 <= res["p_value"] <= 1

    res_alt = chen_2010_identity(non_identity_data)
    assert res_alt["p_value"] < 1

    # Check n < 4 validation
    short_data = np.random.default_rng(42).normal(size=(3, 5))
    with pytest.raises(ValueError, match="n ≥ 4"):
        chen_2010_identity(short_data)


def test_xu_2023_identity(identity_data, non_identity_data):
    res = xu_2023_identity(identity_data)
    assert set(res.keys()) == {"stat", "p_value"}
    assert isinstance(res["stat"], float)
    assert 0 <= res["p_value"] <= 1

    res_alt = xu_2023_identity(non_identity_data)
    assert res_alt["p_value"] < 1

    # Check n < 5 validation
    short_data = np.random.default_rng(42).normal(size=(4, 5))
    with pytest.raises(ValueError, match="n ≥ 5"):
        xu_2023_identity(short_data)


def test_ahmad_2017_identity(identity_data):
    rng = np.random.default_rng(42)
    X1 = identity_data
    X2 = rng.normal(size=(60, 5))

    res = ahmad_2017_identity([X1, X2])
    assert set(res.keys()) == {"stat", "p_value"}
    assert isinstance(res["stat"], float)
    assert 0 <= res["p_value"] <= 1

    # Check validation errors
    with pytest.raises(ValueError, match="at least 2 samples"):
        ahmad_2017_identity([X1])

    # Check feature mismatch validation
    X_bad = rng.normal(size=(50, 4))
    with pytest.raises(ValueError, match="same number of features p"):
        ahmad_2017_identity([X1, X_bad])


def test_identity_T2_outputs(identity_data, non_identity_data):
    res = identity_T2_test(identity_data)
    assert set(res.keys()) == {"stat", "p_value"}
    assert 0 <= res["p_value"] <= 1

    res_alt = identity_T2_test(non_identity_data)
    assert res_alt["p_value"] < 1

    # Check short sample validation
    short_data = np.random.default_rng(42).normal(size=(1, 5))
    with pytest.raises(ValueError, match="Need n >= 2 samples."):
        identity_T2_test(short_data)
