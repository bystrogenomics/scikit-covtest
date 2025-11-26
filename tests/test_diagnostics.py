# tests/test_diagnostics.py
import numpy as np
import pytest

import covtest.diagnostics.assumptions as diag
from covtest.datasets.loader import load_iris
from covtest.diagnostics.assumptions import hz_test, mardia_tests, royston_test


@pytest.fixture
def gaussian_data():
    rng = np.random.default_rng(42)
    return rng.normal(size=(200, 10))


@pytest.fixture
def degenerate_data():
    # Perfect collinearity (two identical features)
    X = np.tile(np.arange(100)[:, None], (1, 2))
    return np.hstack([X, np.random.normal(size=(100, 3))])


def test_eigen_spectrum_returns_dict(gaussian_data):
    res = diag.eigen_spectrum(gaussian_data, plot=False)
    assert isinstance(res, dict)
    assert "eigenvalues" in res
    assert res["eigenvalues"].ndim == 1
    assert len(res["eigenvalues"]) == gaussian_data.shape[1]


def test_condition_and_rank_well_behaved(gaussian_data):
    res = diag.condition_and_rank(gaussian_data)
    assert "condition_number" in res
    assert res["numerical_rank"] <= gaussian_data.shape[1]
    assert res["effective_rank"] <= gaussian_data.shape[1]
    assert isinstance(res["warnings"], list)


def test_condition_and_rank_detects_degeneracy(degenerate_data):
    res = diag.condition_and_rank(degenerate_data)
    # Should issue at least one warning about rank deficiency
    assert any("rank" in w.lower() for w in res["warnings"])


def test_mardia_against_mvn():
    X, y = load_iris(return_X_y=True)
    setosa = X[y == 0]
    res = mardia_tests(setosa)
    assert np.abs(res["stat"][0] - 25.664345) < 1e-4
    assert np.abs(res["stat"][1] - 1.294992) < 1e-4

    assert np.abs(res["p_value"][0] - 0.1771859) < 1e-4
    assert np.abs(res["p_value"][1] - 0.1953229) < 1e-4


def test_royston_mvn():
    X, y = load_iris(return_X_y=True)
    setosa = X[y == 0]
    res = royston_test(setosa)
    assert np.abs(res["stat"] - 31.21) < 1e-1
    assert np.abs(res["p_value"] - 2.5e-6) < 1e-5


def test_hz_mvn():
    X, y = load_iris(return_X_y=True)
    setosa = X[y == 0]
    res = hz_test(setosa)
    assert np.abs(res["stat"] - 0.948845) < 1e-4
    assert np.abs(res["p_value"] - 0.04995356) < 1e-4
