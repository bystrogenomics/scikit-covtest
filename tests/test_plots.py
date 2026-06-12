# test_pval_plots.py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

# Import your plotting functions
from covtest.plotting.null import (
    plot_pval_calibration,
    plot_pval_cumdev,
    plot_pval_ecdf,
    plot_pval_histogram,
    plot_pval_obs_vs_exp,
    plot_pval_qq,
    plot_pval_running_mean,
    plot_pval_survival,
    plot_pval_zdist,
    plot_pvalue_diagnostics_grid,
)

plt.show = lambda *args, **kwargs: None
matplotlib.use("Agg")


@pytest.fixture
def sample_pvals():
    rng = np.random.default_rng(0)
    return rng.uniform(0, 1, size=200)


# --- Individual plotting functions ---


@pytest.mark.parametrize(
    "plot_func",
    [
        plot_pval_histogram,
        plot_pval_qq,
        plot_pval_ecdf,
        plot_pval_cumdev,
        plot_pval_obs_vs_exp,
        plot_pval_survival,
        plot_pval_running_mean,
        plot_pval_zdist,
        plot_pval_calibration,
    ],
)
def test_plot_functions_run_no_ax(sample_pvals, plot_func):
    """Ensure plotting functions run without error when ax=None."""
    plot_func(sample_pvals)  # should create its own fig/ax
    plt.close("all")


@pytest.mark.parametrize(
    "plot_func",
    [
        plot_pval_histogram,
        plot_pval_qq,
        plot_pval_ecdf,
        plot_pval_cumdev,
        plot_pval_obs_vs_exp,
        plot_pval_survival,
        plot_pval_running_mean,
        plot_pval_zdist,
        plot_pval_calibration,
    ],
)
def test_plot_functions_with_ax(sample_pvals, plot_func):
    """Ensure plotting functions can draw into provided axis."""
    fig, ax = plt.subplots()
    plot_func(sample_pvals, ax=ax)
    plt.close(fig)


# --- Combined grid ---


def test_plot_grid_runs(sample_pvals):
    """Ensure the combined grid runs without error."""
    plot_pvalue_diagnostics_grid(sample_pvals, n_bins=5)
    plt.close("all")
