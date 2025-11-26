import numpy as np

from covtest.multiplicity import fwer


def test_bonferroni_simple():
    pvals = np.array([0.01, 0.2, 0.5])
    res = fwer.bonferroni(pvals, alpha=0.05)
    assert np.allclose(res["pvals_adj"], [0.03, 0.6, 1.0])
    assert bool(res["rejected"][0]) is True
    assert bool(res["rejected"][1]) is False


def test_holm_ordering():
    pvals = np.array([0.01, 0.04, 0.2])
    res = fwer.holm(pvals, alpha=0.05)
    # the smallest p-value should be rejected
    assert bool(res["rejected"][np.argmin(pvals)]) is True
    pvals = np.array([0.001, 0.01, 0.2, 0.5])
    pvals2 = np.array([0.001, 0.01, 0.2, 0.5, 0.99])
    res = fwer.holm(pvals, alpha=0.05)
    true_p = np.array([0.004, 0.03, 0.4, 0.5])
    assert np.allclose(res["pvals_adj"], true_p)
    true_p2 = np.array([0.005, 0.04, 0.6, 1.0, 1.0])
    res = fwer.holm(pvals2, alpha=0.05)
    assert np.allclose(res["pvals_adj"], true_p2)


def test_hochberg_behavior():
    pvals = np.array([0.001, 0.01, 0.2, 0.5])
    pvals2 = np.array([0.001, 0.01, 0.2, 0.5, 0.99])
    res = fwer.hochberg(pvals, alpha=0.05)
    true_p = np.array([0.004, 0.03, 0.4, 0.5])
    assert np.allclose(res["pvals_adj"], true_p)
    true_p2 = np.array([0.005, 0.04, 0.6, 0.990, 0.990])
    res = fwer.hochberg(pvals2, alpha=0.05)
    assert np.allclose(res["pvals_adj"], true_p2)


def test_hommel_trivial():
    pvals = np.array([0.001, 0.01, 0.2, 0.5])
    pvals2 = np.array([0.001, 0.01, 0.2, 0.5, 0.99])
    res = fwer.hochberg(pvals, alpha=0.05)
    true_p = np.array([0.004, 0.03, 0.4, 0.5])
    assert np.allclose(res["pvals_adj"], true_p)
    true_p2 = np.array([0.005, 0.04, 0.6, 0.990, 0.990])
    res = fwer.hochberg(pvals2, alpha=0.05)
    assert np.allclose(res["pvals_adj"], true_p2)


def test_romano_wolf_maxT():
    T_obs = np.array([2.5, 1.8, 3.2])
    rng = np.random.default_rng(0)
    T_boot = rng.standard_normal((200, 3))
    res = fwer.romano_wolf_maxT(T_obs, T_boot, alpha=0.05)
    assert "pvals_adj" in res
    assert res["pvals_adj"].shape == (3,)
    assert res["rejected"].dtype == bool
