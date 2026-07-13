import numpy as np
import pytest

from covtest.methods.hypothesis_spherical import (
    _john_stat,
    bartlett_sphericity_test,
    czz_sphericity_test,
    fisher_2010_sphericity_test,
    hu_2019_sphericity_test,
    john_sphericity,
    srivastava_2005_sphericity,
    srivastava_2014_sphericity_test,
    xu_2023_sphericity_test,
    ahmad2015_sphericity_test,
    muirhead_sphericity_lrt,
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


def test_srivastava_output(identity_data):
    res = srivastava_2005_sphericity(identity_data)
    assert "stat" in res and "p_value" in res
    assert isinstance(res["stat"], float)
    assert 0 <= res["p_value"] <= 1


def test_non_spherical_data_gives_signal(non_spherical_data):
    john_res = john_sphericity(non_spherical_data)
    czz_res = czz_sphericity_test(non_spherical_data)
    sriv_14_res = srivastava_2014_sphericity_test(non_spherical_data)
    fish_10_res = fisher_2010_sphericity_test(non_spherical_data)
    hu_19_res = hu_2019_sphericity_test(non_spherical_data)
    xu_23_res = xu_2023_sphericity_test(non_spherical_data)

    # Expect evidence against null, so p-value should not be 1
    thresh = 0.1
    assert john_res["p_value"] < thresh
    assert czz_res["p_value"] < thresh
    assert sriv_14_res["p_value"] < thresh
    assert fish_10_res["p_value"] < thresh
    assert hu_19_res["p_value"] < thresh
    assert xu_23_res["p_value"] < thresh


def test_srivastava_2014_output(identity_data):
    res = srivastava_2014_sphericity_test(identity_data)
    assert set(res.keys()) == {"stat", "p_value"}
    assert isinstance(res["stat"], float)
    assert 0 <= res["p_value"] <= 1


def test_srivastava_2014_rejects_small_N():
    X = np.random.normal(size=(3, 2))
    with pytest.raises(ValueError, match="requires N >= 4"):
        srivastava_2014_sphericity_test(X)


def test_fisher_2010_output(identity_data):
    res = fisher_2010_sphericity_test(identity_data)
    assert set(res.keys()) == {"stat", "p_value"}
    assert isinstance(res["stat"], float)
    assert 0 <= res["p_value"] <= 1

    res_no_center = fisher_2010_sphericity_test(identity_data, center=False)
    assert set(res_no_center.keys()) == {"stat", "p_value"}
    assert isinstance(res_no_center["stat"], float)
    assert 0 <= res_no_center["p_value"] <= 1


def test_fisher_2010_rejects_small_N():
    X = np.random.normal(size=(7, 2))
    with pytest.raises(ValueError, match="requires N >= 8"):
        fisher_2010_sphericity_test(X)


def test_hu_2019_output(identity_data):
    res = hu_2019_sphericity_test(identity_data)
    assert set(res.keys()) == {"stat", "p_value"}
    assert isinstance(res["stat"], float)
    assert 0 <= res["p_value"] <= 1

    res_all = hu_2019_sphericity_test(identity_data, return_all=True)
    assert set(res_all.keys()) == {"T1", "T2", "Tm"}
    for key in ("T1", "T2", "Tm"):
        assert set(res_all[key].keys()) == {"stat", "p_value"}
        assert isinstance(res_all[key]["stat"], float)
        assert 0 <= res_all[key]["p_value"] <= 1


def test_hu_2019_rejects_small_n():
    X = np.random.normal(size=(1, 2))
    with pytest.raises(ValueError, match="requires n >= 2"):
        hu_2019_sphericity_test(X)


def test_xu_2023_output(identity_data):
    res = xu_2023_sphericity_test(identity_data)
    assert set(res.keys()) == {"stat", "p_value"}
    assert isinstance(res["stat"], float)
    assert 0 <= res["p_value"] <= 1

    res_center = xu_2023_sphericity_test(identity_data, center=True)
    assert set(res_center.keys()) == {"stat", "p_value"}
    assert isinstance(res_center["stat"], float)
    assert 0 <= res_center["p_value"] <= 1


def test_xu_2023_rejects_small_n():
    X = np.random.normal(size=(4, 2))
    with pytest.raises(ValueError, match="requires n >= 5"):
        xu_2023_sphericity_test(X)


def test_ahmad2015_sphericity_test(identity_data, non_spherical_data):
    res = ahmad2015_sphericity_test(identity_data)
    assert set(res.keys()) == {"stat", "p_value"}
    assert isinstance(res["stat"], float)
    assert 0 <= res["p_value"] <= 1

    res_alt = ahmad2015_sphericity_test(non_spherical_data)
    assert res_alt["p_value"] < 1


def test_muirhead_sphericity_lrt(identity_data, non_spherical_data):
    res = muirhead_sphericity_lrt(identity_data)
    assert set(res.keys()) == {"stat", "p_value"}
    assert isinstance(res["stat"], float)
    assert 0 <= res["p_value"] <= 1

    res_alt = muirhead_sphericity_lrt(non_spherical_data)
    assert res_alt["p_value"] < 1

    # Check S-input variant
    n, p = identity_data.shape
    S = np.cov(identity_data, rowvar=False)
    res_s = muirhead_sphericity_lrt(S=S, n=n)
    assert set(res_s.keys()) == {"stat", "p_value"}
    assert isinstance(res_s["stat"], float)
    assert 0 <= res_s["p_value"] <= 1

    # Validation error check
    with pytest.raises(ValueError, match="Provide either X or S"):
        muirhead_sphericity_lrt()


def test_standardize_T_values():
    from covtest.methods._ahmad import _standardize_T

    # Test values: T=0.01, n=1000, p=80
    # Expected: z = (n / 2.0) * T = (1000 / 2) * 0.01 = 5.0
    z_auto, cal_auto = _standardize_T(T=0.01, n=1000, p=80, calibration="auto")
    assert np.isclose(z_auto, 5.0)
    assert cal_auto == "ahmad2015"

    z_ahmad, cal_ahmad = _standardize_T(T=0.01, n=1000, p=80, calibration="ahmad2015")
    assert np.isclose(z_ahmad, 5.0)
    assert cal_ahmad == "ahmad2015"

    z_large, cal_large = _standardize_T(
        T=0.01, n=1000, p=80, calibration="large_p_small_n"
    )
    assert np.isclose(z_large, 5.0)
    assert cal_large == "ahmad2015"

    # Verify ratio calibration is only when explicitly requested
    z_ratio, cal_ratio = _standardize_T(T=0.01, n=1000, p=80, calibration="ratio")
    # For ratio: c = p/n = 80/1000 = 0.08
    # var_nT = 4.0 * (1.0 + 2.0 / c) = 4 * (1 + 25) = 104
    # Expected: (n * T) / sqrt(var_nT) = 10 / sqrt(104) = 0.98058
    assert np.isclose(z_ratio, 10.0 / np.sqrt(104.0))
    assert cal_ratio == "ratio"

    # Verify invalid calibration raises ValueError
    with pytest.raises(ValueError, match="calibration must be one of"):
        _standardize_T(T=0.01, n=1000, p=80, calibration="invalid")


def test_ahmad2015_sphericity_calibration_simulation():
    rng = np.random.default_rng(42)
    n, p = 1000, 80
    num_reps = 100
    z_scores = []

    # Generate null data: standard Gaussian covariance = I
    for _ in range(num_reps):
        X = rng.normal(size=(n, p))
        res = ahmad2015_sphericity_test(X, center=False, calibration="auto")
        # Under null, (n/2)*T -> N(0,1)
        T1 = res["stat"]
        z = (n / 2.0) * T1
        z_scores.append(z)

    z_scores = np.array(z_scores)
    z_std = np.std(z_scores)
    # Check that z-statistics are not artificially compressed near zero
    # Under the ratio calibration it would be compressed to ~0.2,
    # under the correct calibration the standard deviation should be close to 1.0 (e.g. > 0.7)
    assert z_std > 0.7
