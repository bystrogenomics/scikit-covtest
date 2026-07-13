import numpy as np
import pytest

from covtest.methods.hypothesis_two_sample import (
    _schott_2001_two_sample_stat,
    _srivastava_2007_stat,
    _srivastava_yanagihara_stat,
    boxm_test,
    cai_2013_two_sample,
    cai_liu_xia_2013_two_sample_test,
    chang2016,
    chang_2017_perturbation_max_test,
    schott_2001,
    srivastava_two_sample_2007,
    srivastava_yanagihara_two_sample,
    two_sample_cov_test,
    tyler_two_sample,
    wald_two_sample,
)


@pytest.fixture
def data_two_groups():
    rng = np.random.default_rng(0)
    X1 = rng.normal(size=(30, 5))
    X2 = rng.normal(size=(35, 5))
    return X1, X2


@pytest.fixture
def data_three_groups():
    rng = np.random.default_rng(1)
    return {
        "g1": rng.normal(size=(25, 4)),
        "g2": rng.normal(size=(30, 4)),
        "g3": rng.normal(size=(28, 4)),
    }


def test_boxm_test_chisq_and_F(data_two_groups):
    X1, X2 = data_two_groups
    res_chi = boxm_test(X1, X2, type="chi.squared")
    res_f = boxm_test(X1, X2, type="F")
    assert set(res_chi.keys()) == {"stat", "p_value"}
    assert set(res_f.keys()) == {"stat", "p_value"}
    assert 0 <= res_chi["p_value"] <= 1
    assert 0 <= res_f["p_value"] <= 1


def test_boxm_test_invalid_type(data_two_groups):
    X1, X2 = data_two_groups
    with pytest.raises(ValueError):
        boxm_test(X1, X2, type="invalid")


def test_schott_stat_and_test(data_three_groups):
    matrices = list(data_three_groups.values())
    stat = _schott_2001_two_sample_stat(matrices)
    assert isinstance(stat, float)
    res = schott_2001(data_three_groups["g1"], data_three_groups["g2"])
    assert set(res.keys()) == {"stat", "p_value"}


def test_srivastava_yanagihara(data_three_groups):
    matrices = list(data_three_groups.values())
    stat = _srivastava_yanagihara_stat(matrices)
    assert isinstance(stat, float)
    res = srivastava_yanagihara_two_sample(matrices[0], matrices[1])
    assert set(res.keys()) == {"stat", "p_value"}
    assert 0 <= res["p_value"] <= 1


def test_srivastava_2007(data_three_groups):
    matrices = list(data_three_groups.values())
    stat = _srivastava_2007_stat(matrices)
    assert isinstance(stat, float)
    res = srivastava_two_sample_2007(
        data_three_groups["g1"], data_three_groups["g2"]
    )
    assert set(res.keys()) == {"stat", "p_value"}
    assert 0 <= res["p_value"] <= 1


def test_wald_two_sample(data_two_groups):
    X1, X2 = data_two_groups
    res = wald_two_sample(X1, X2)
    assert set(res.keys()) == {"stat", "p_value"}
    assert 0 <= res["p_value"] <= 1


def test_wald_dimension_mismatch():
    X1 = np.random.normal(size=(20, 4))
    X2 = np.random.normal(size=(25, 5))  # mismatched p
    with pytest.raises(ValueError):
        wald_two_sample(X1, X2)


def test_tyler_two_sample(data_two_groups):
    X1, X2 = data_two_groups
    res_known = tyler_two_sample(X1, X2, unknown_mean=False)
    res_unknown = tyler_two_sample(X1, X2, unknown_mean=True)
    assert set(res_known.keys()) == {"stat", "p_value"}
    assert set(res_unknown.keys()) == {"stat", "p_value"}
    assert 0 <= res_known["p_value"] <= 1
    assert 0 <= res_unknown["p_value"] <= 1


def test_two_sample_cov_test(data_two_groups):
    X1, X2 = data_two_groups
    # Needs matching dimensions
    res = two_sample_cov_test(X1, X2)
    assert set(res.keys()) == {"p_value", "z_value", "lrt"}
    assert 0 <= res["p_value"] <= 1


def test_cai_liu_xia_2013(data_two_groups):
    X1, X2 = data_two_groups
    res = cai_liu_xia_2013_two_sample_test(X1, X2)
    assert "Mn" in res
    assert 0 <= res["p_value"] <= 1


def test_chang_2017(data_two_groups):
    X1, X2 = data_two_groups
    res = chang_2017_perturbation_max_test(X1, X2, B=10)
    assert "Tmax" in res
    assert 0 <= res["p_value"] <= 1


def test_cai_2013_two_sample(data_two_groups):
    X1, X2 = data_two_groups
    res = cai_2013_two_sample(X1, X2)
    assert set(res.keys()) == {"stat", "p_value"}
    assert 0 <= res["p_value"] <= 1


def test_chang2016(data_two_groups):
    X1, X2 = data_two_groups
    res = chang2016(X1, X2, J=10)
    assert set(res.keys()) == {"stat", "p_value"}
    assert 0 <= res["p_value"] <= 1
