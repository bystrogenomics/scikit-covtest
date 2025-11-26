import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def benchmark_computation_dimension(
    methods,
    N,
    p_list,
    n_reps=5,
    mark_progress=False,
    rng=None,
    two_sample=False,
):
    """
    Benchmark computation time of multiple methods across different dimensions.

    Parameters
    ----------
    methods : dict[str, callable]
        Dictionary mapping method name -> function.
        Each function must accept a 2D array X (N x p).
    N : int
        Number of samples (rows).
    p_list : list[int]
        List of dimensionalities (number of features) to test.
    n_reps : int, default=5
        Number of repetitions per (method, p) to average computation time.
    rng : np.random.Generator, optional
        Random generator for reproducibility.

    Returns
    -------
    results : pd.DataFrame
        DataFrame with columns [method, p, mean_time, std_time].
    """
    if rng is None:
        rng = np.random.default_rng(0)

    records = []
    for p in p_list:
        # Generate data once per dimension for fairness
        if mark_progress:
            print(N)
        X = rng.normal(size=(N, p))
        if two_sample:
            Y = rng.normal(size=(N, p))

        for method_name, func in methods.items():
            times = []
            for _ in range(n_reps):
                t0 = time.perf_counter()
                if two_sample:
                    _ = func(X, Y)
                else:
                    _ = func(X)
                t1 = time.perf_counter()
                times.append(t1 - t0)
            mean_t = np.mean(times)
            std_t = np.std(times)
            records.append(
                {
                    "method": method_name,
                    "p": p,
                    "mean_time": mean_t,
                    "std_time": std_t,
                }
            )

    return pd.DataFrame.from_records(records)


def benchmark_computation_sample_size(
    methods,
    N_list,
    p,
    n_reps=5,
    mark_progress=False,
    rng=None,
    two_sample=False,
):
    """
    Benchmark computation time of multiple methods across different dimensions.

    Parameters
    ----------
    methods : dict[str, callable]
        Dictionary mapping method name -> function.
        Each function must accept a 2D array X (N x p).
    N : int
        Number of samples (rows).
    p_list : list[int]
        List of dimensionalities (number of features) to test.
    n_reps : int, default=5
        Number of repetitions per (method, p) to average computation time.
    rng : np.random.Generator, optional
        Random generator for reproducibility.

    Returns
    -------
    results : pd.DataFrame
        DataFrame with columns [method, p, mean_time, std_time].
    """
    if rng is None:
        rng = np.random.default_rng(0)

    records = []
    for N in N_list:
        # Generate data once per dimension for fairness
        if mark_progress:
            print(N)

        X = rng.normal(size=(N, p))
        if two_sample:
            Y = rng.normal(size=(N, p))

        for method_name, func in methods.items():
            times = []
            for _ in range(n_reps):
                t0 = time.perf_counter()
                if two_sample:
                    _ = func(X, Y)
                else:
                    _ = func(X)
                t1 = time.perf_counter()
                times.append(t1 - t0)
            mean_t = np.mean(times)
            std_t = np.std(times)
            records.append(
                {
                    "method": method_name,
                    "N": N,
                    "mean_time": mean_t,
                    "std_time": std_t,
                }
            )

    return pd.DataFrame.from_records(records)


def plot_computation_dimension(results, title=None, savename=None):
    pastel_colors = [
        "#AEC6CF",  # pastel blue
        "#FFB347",  # pastel orange
        "#77DD77",  # pastel green
        "#FFD1DC",  # pastel pink
        "#CBAACB",  # pastel purple
        "#FFFACD",  # pastel yellow
        "#B39EB5",  # pastel violet
        "#FF6961",  # pastel red
        "#03C03C",  # pastel teal
        "#FDFD96",  # pastel lemon
        "#779ECB",  # pastel sky blue
        "#966FD6",  # pastel lavender
        "#CB99C9",  # pastel magenta
        "#CFCFC4",  # pastel gray
        "#F49AC2",  # pastel rose
    ]

    fig, ax = plt.subplots(figsize=(6, 4))
    for i, method in enumerate(results["method"].unique()):
        df = results[results["method"] == method]
        color = pastel_colors[i % len(pastel_colors)]
        ax.errorbar(
            df["p"],
            df["mean_time"],
            yerr=df["std_time"],
            label=method,
            color=color,
            ecolor=color,
            marker="o",
            alpha=0.6,
            linewidth=2,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Dimension p", fontsize=20)
    ax.set_ylabel("Computation time (s)", fontsize=20)
    if title is None:
        title = "Benchmarking Dimensional Complexity"
    ax.set_title(
        title,
        fontsize=14,
        fontweight="bold",
    )
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)
    plt.tight_layout()
    if savename is not None:
        plt.savefig(savename)
    plt.show()


def plot_computation_sample_size(results, title=None, savename=None):
    pastel_colors = [
        "#AEC6CF",  # pastel blue
        "#FFB347",  # pastel orange
        "#77DD77",  # pastel green
        "#FFD1DC",  # pastel pink
        "#CBAACB",  # pastel purple
        "#FFFACD",  # pastel yellow
        "#B39EB5",  # pastel violet
        "#FF6961",  # pastel red
        "#03C03C",  # pastel teal
        "#FDFD96",  # pastel lemon
        "#779ECB",  # pastel sky blue
        "#966FD6",  # pastel lavender
        "#CB99C9",  # pastel magenta
        "#CFCFC4",  # pastel gray
        "#F49AC2",  # pastel rose
    ]

    fig, ax = plt.subplots(figsize=(6, 4))
    for i, method in enumerate(results["method"].unique()):
        df = results[results["method"] == method]
        color = pastel_colors[i % len(pastel_colors)]
        ax.errorbar(
            df["N"],
            df["mean_time"],
            yerr=df["std_time"],
            label=method,
            color=color,
            ecolor=color,
            marker="o",
            alpha=0.6,
            linewidth=2,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Sample Size N", fontsize=20)
    ax.set_ylabel("Computation time (s)", fontsize=20)
    if title is None:
        title = "Benchmarking Sample Complexity"
    ax.set_title(
        title,
        fontsize=24,
    )
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)
    plt.tight_layout()
    if savename is not None:
        plt.savefig(savename)
    plt.show()
