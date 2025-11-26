import numpy as np
import pandas as pd
from covtest.testing.timing import (
    benchmark_computation_dimension,
    benchmark_computation_sample_size,
    plot_computation_dimension,
    plot_computation_sample_size,
)


def dummy_method(X):
    return np.mean(X)


def dummy_method_two_sample(X, Y):
    return np.mean(X) + np.mean(Y)


def test_benchmark_computation_dimension():
    methods = {"dummy": dummy_method}
    N = 10
    p_list = [5, 10]
    results = benchmark_computation_dimension(methods, N, p_list, n_reps=2)

    assert isinstance(results, pd.DataFrame)
    assert len(results) == 2
    assert set(results.columns) == {"method", "p", "mean_time", "std_time"}
    assert results["p"].tolist() == p_list


def test_benchmark_computation_dimension_two_sample():
    methods = {"dummy": dummy_method_two_sample}
    N = 10
    p_list = [5, 10]
    results = benchmark_computation_dimension(
        methods, N, p_list, n_reps=2, two_sample=True
    )

    assert isinstance(results, pd.DataFrame)
    assert len(results) == 2


def test_benchmark_computation_sample_size():
    methods = {"dummy": dummy_method}
    N_list = [10, 20]
    p = 5
    results = benchmark_computation_sample_size(methods, N_list, p, n_reps=2)

    assert isinstance(results, pd.DataFrame)
    assert len(results) == 2
    assert set(results.columns) == {"method", "N", "mean_time", "std_time"}
    assert results["N"].tolist() == N_list


def test_benchmark_computation_sample_size_two_sample():
    methods = {"dummy": dummy_method_two_sample}
    N_list = [10, 20]
    p = 5
    results = benchmark_computation_sample_size(
        methods, N_list, p, n_reps=2, two_sample=True
    )

    assert isinstance(results, pd.DataFrame)
    assert len(results) == 2


def test_plot_computation_dimension(tmp_path):
    methods = {"dummy": dummy_method}
    N = 10
    p_list = [5, 10]
    results = benchmark_computation_dimension(methods, N, p_list, n_reps=2)

    # Test plotting without saving
    plot_computation_dimension(results)

    # Test plotting with saving
    save_path = tmp_path / "plot_dim.png"
    plot_computation_dimension(results, savename=str(save_path))
    assert save_path.exists()


def test_plot_computation_sample_size(tmp_path):
    methods = {"dummy": dummy_method}
    N_list = [10, 20]
    p = 5
    results = benchmark_computation_sample_size(methods, N_list, p, n_reps=2)

    # Test plotting without saving
    plot_computation_sample_size(results)

    # Test plotting with saving
    save_path = tmp_path / "plot_sample.png"
    plot_computation_sample_size(results, savename=str(save_path))
    assert save_path.exists()
