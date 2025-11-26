from covtest.multiplicity import online


def test_lord_sequence():
    pvals = [0.001, 0.5, 0.01, 0.8]
    res = online.lord(pvals, alpha=0.1)
    assert len(res["rejected"]) == len(pvals)
    assert any(res["rejected"])  # should reject at least one


def test_saffron_sequence():
    pvals = [0.001, 0.5, 0.01, 0.8]
    res = online.saffron(pvals, alpha=0.1, lambda0=0.5)
    assert len(res["rejected"]) == len(pvals)
    # at least the very small p-value should be rejected
    assert res["rejected"][0]
