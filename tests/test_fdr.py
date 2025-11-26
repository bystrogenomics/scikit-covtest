import numpy as np

from covtest.multiplicity import fdr


def test_bh_basic():
    pvals = np.array([0.001, 0.01, 0.2, 0.5])
    res = fdr.benjamini_hochberg(pvals, alpha=0.05)
    # smallest two should be rejected
    assert res["rejected"][0]
    assert res["rejected"][1]
    true_q = np.array([0.004, 0.02, 0.2666666667, 0.5])
    assert np.allclose(res["qvals"], true_q)

    pvals2 = np.array([0.001, 0.01, 0.2, 0.5, 0.99])
    true_q2 = np.array([0.005, 0.025, 0.3333333333, 0.625, 0.99])
    res2 = fdr.benjamini_hochberg(pvals2, alpha=0.05)
    assert np.allclose(res2["qvals"], true_q2)


def test_bl():
    pvals = np.array([0.001, 0.01, 0.2, 0.5])
    res = fdr.benjamini_liu(pvals, alpha=0.05)
    # smallest two should be rejected
    assert res["rejected"][0]
    assert res["rejected"][1]
    true_q = np.array([0.003994004, 0.02227575, 0.18, 0.18])
    assert np.allclose(res["qvals"], true_q)

    pvals2 = np.array([0.001, 0.01, 0.2, 0.5, 0.99])
    true_q2 = np.array([0.00499001, 0.03152319, 0.2928, 0.3, 0.3])
    res2 = fdr.benjamini_liu(pvals2, alpha=0.05)
    assert np.allclose(res2["qvals"], true_q2)


def test_by_more_conservative():
    pvals = np.array([0.001, 0.01, 0.2, 0.5])
    bh = fdr.benjamini_hochberg(pvals, alpha=0.05)
    by = fdr.benjamini_yekutieli(pvals, alpha=0.05)
    # BY q-values should be >= BH q-values
    assert np.all(by["qvals"] >= bh["qvals"])

    true_q = np.array([0.008333333, 0.04166666667, 0.55555555, 1.0])
    assert np.allclose(by["qvals"], true_q)

    pvals2 = np.array([0.001, 0.01, 0.2, 0.5, 0.99])
    true_q2 = np.array([0.01141667, 0.05708333, 0.761111111, 1.0, 1.0])
    res2 = fdr.benjamini_yekutieli(pvals2, alpha=0.05)
    assert np.allclose(res2["qvals"], true_q2)


def test_blaroq():
    pvals = np.array([0.001, 0.01, 0.2, 0.5])
    res = fdr.blaroq(pvals, alpha=0.05)
    # smallest two should be rejected
    assert res["rejected"][0]
    assert res["rejected"][1]
    true_q = np.array([0.00492515, 0.03574775, 0.66342122, 1.0])
    assert np.allclose(res["qvals"], true_q)

    pvals2 = np.array([0.001, 0.01, 0.2, 0.5, 0.99])
    true_q2 = np.array([0.006781121, 0.044402477, 0.781395082, 1.0, 1.0])
    res2 = fdr.blaroq(pvals2, alpha=0.05)
    assert np.allclose(res2["qvals"], true_q2)


def test_weighted_bh():
    pvals = np.array([0.01, 0.2, 0.5])
    weights = np.array([2.0, 1.0, 1.0])
    res = fdr.weighted_bh(pvals, weights, alpha=0.05)
    # smallest p-value with higher weight should be rejected
    assert res["rejected"][0]


def test_storey_qvalues_pi0_estimation():
    pvals = np.linspace(0.0, 1.0, 100)
    res = fdr.storey_qvalues(pvals, alpha=0.1)
    assert 0 <= res["pi0"] <= 1
    assert res["qvals"].shape == (100,)
