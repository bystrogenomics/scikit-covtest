import numpy as np
import pandas as pd
import pytest

from covtest.methods.hypothesis_spherical import (
    john_sphericity,
    sk_test,
    srivastava_2005_sphericity,
)
from covtest.testing.timing import (
    benchmark_computation_dimension,
    benchmark_computation_sample_size,
)


@pytest.fixture
def small_data():
    rng = np.random.default_rng(0)
    return rng.normal(size=(30, 10))  # small test matrix


@pytest.fixture
def methods():
    return {
        "john_sphericity": john_sphericity,
        "srivastava_2005_sphericity": srivastava_2005_sphericity,
        "sk_test": sk_test,
    }


def test_benchmark_pipeline_runs(methods):
    N = 30
    p_list = [5, 10]
    results = benchmark_computation_dimension(methods, N, p_list, n_reps=2)

    # Check output type
    assert isinstance(results, pd.DataFrame)

    # Required columns
    required_cols = {"method", "p", "mean_time", "std_time"}
    assert required_cols.issubset(results.columns)

    # Each method should appear
    for m in methods.keys():
        assert m in results["method"].values

    # Each p in p_list should appear
    for p in p_list:
        assert p in results["p"].values

    # Times should be non-negative
    assert (results["mean_time"] >= 0).all()
    assert (results["std_time"] >= 0).all()


def test_benchmark_pipeline_runs_sample_size(methods):
    p = 30
    N_list = [50, 100]
    results = benchmark_computation_sample_size(methods, N_list, p, n_reps=2)

    # Check output type
    assert isinstance(results, pd.DataFrame)

    # Required columns for sample-size benchmarking
    required_cols = {"method", "N", "mean_time", "std_time"}
    assert required_cols.issubset(results.columns)

    # Each method should appear
    for m in methods.keys():
        assert m in results["method"].values

    # Each N in N_list should appear
    for N in N_list:
        assert N in results["N"].values

    # Times should be non-negative
    assert (results["mean_time"] >= 0).all()
    assert (results["std_time"] >= 0).all()
