import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

_c1 = "dodgerblue"
_c2 = "grey"
fs1 = 20
fs2 = 24
fs3 = 16
size1 = 6
size2 = 4


def _clean_pvals(pvals, eps=0.0):
    """Return finite p-values clipped to [eps, 1 - eps]."""
    p = np.asarray(pvals, dtype=float)
    p = p[np.isfinite(p)]
    if p.size == 0:
        return p
    hi = 1.0 - np.finfo(float).eps
    return np.clip(p, eps, hi)


# ---- Styling helper ----
def _style_axes(ax, title):
    ax.set_title(title, fontsize=fs2, fontweight="bold")
    ax.tick_params(axis="both", which="major", labelsize=12)
    # Remove top/right spines, thicken bottom/left
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)


# ---- 1. Histogram ----
def plot_pval_histogram(pvals, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(size1, size2))

    ax.hist(pvals, bins=20, density=True, alpha=0.7, color=_c1, edgecolor="k")
    ax.hlines(
        1, 0, 1, colors=_c2, linestyles="--", linewidth=2, label="Uniform(0,1)"
    )
    ax.set_xlim(0, 1)
    ax.set_xlabel("p-value", fontsize=fs1)
    ax.set_ylabel("Density", fontsize=fs1)
    ax.legend(fontsize=11)
    _style_axes(ax, "Histogram of p-values")
    plt.tight_layout()
    plt.show()


# ---- 2. Q-Q Plot ----
def plot_pval_qq(pvals, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(size1, size2))

    pvals = np.sort(pvals)
    n = len(pvals)
    exp_q = np.linspace(0, 1, n, endpoint=False) + 0.5 / n
    ax.scatter(exp_q, pvals, s=15, alpha=0.6, color=_c1)
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=2, color=_c2)
    ax.set_xlabel("Expected quantiles", fontsize=fs1)
    ax.set_ylabel("Observed quantiles", fontsize=fs1)
    _style_axes(ax, "Q–Q Plot vs Uniform")
    plt.tight_layout()
    plt.show()


# ---- 3. ECDF / P–P ----
def plot_pval_ecdf(pvals, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(size1, size2))

    pvals = np.sort(pvals)
    n = len(pvals)
    ecdf_y = np.arange(1, n + 1) / n
    ax.plot(pvals, ecdf_y, label="Empirical CDF", color=_c1, linewidth=2)
    ax.plot([0, 1], [0, 1], "--", linewidth=2, label="Uniform(0,1)", color=_c2)
    ax.set_xlabel("p-value", fontsize=fs1)
    ax.set_ylabel("ECDF", fontsize=fs1)
    ax.legend(fontsize=11)
    _style_axes(ax, "Empirical CDF")
    plt.tight_layout()
    plt.show()


# ---- 4. Cumulative deviation ----
def plot_pval_cumdev(pvals, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(size1, size2))

    pvals = np.sort(pvals)
    n = len(pvals)
    ecdf_y = np.arange(1, n + 1) / n
    cum_dev = ecdf_y - pvals
    ax.plot(pvals, cum_dev, color=_c1, linewidth=2)
    ax.axhline(0, color=_c2, linestyle="--", linewidth=2)
    ax.set_xlabel("p-value", fontsize=fs1)
    ax.set_ylabel("Obs - Exp", fontsize=fs1)
    _style_axes(ax, "Cumulative deviation plot")
    plt.tight_layout()
    plt.show()


# ---- 5. Observed vs expected counts ----
def plot_pval_obs_vs_exp(pvals, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(size1, size2))

    n = len(pvals)
    thresholds = np.logspace(-4, 0, 20)
    obs = np.array([(pvals <= t).sum() for t in thresholds])
    exp = n * thresholds
    ax.plot(exp, obs, "o-", color=_c1, linewidth=2)
    ax.plot([1, n], [1, n], "--", linewidth=2, color=_c2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Expected # ≤ t", fontsize=fs1)
    ax.set_ylabel("Observed # ≤ t", fontsize=fs1)
    _style_axes(ax, "Obs. vs Expect. counts")
    plt.tight_layout()
    plt.show()


# ---- 6. Survival function ----
def plot_pval_survival(pvals, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(size1, size2))

    pvals = np.sort(pvals)
    n = len(pvals)
    ecdf_y = np.arange(1, n + 1) / n
    surv_obs = 1 - ecdf_y
    ax.plot(pvals, surv_obs, label="Observed", color=_c1, linewidth=2)
    ax.plot(
        pvals, 1 - pvals, "--", linewidth=2, label="Uniform(0,1)", color=_c2
    )
    ax.set_yscale("log")
    ax.set_xlabel("p-value", fontsize=fs1)
    ax.set_ylabel("Survival probability", fontsize=fs1)
    ax.legend(fontsize=11)
    _style_axes(ax, "Survival function")
    plt.tight_layout()
    plt.show()


# ---- 7. Running mean ----
def plot_pval_running_mean(pvals, ax=None):
    # Clean but allow zeros (eps=0) since no transform to z is used here
    p = _clean_pvals(pvals, eps=0.0)
    if p.size == 0:
        if ax is None:
            fig, ax = plt.subplots(figsize=(size1, size2))
        ax.text(0.5, 0.5, "No finite p-values", ha="center", va="center")
        _style_axes(ax, "Running mean plot")
        plt.tight_layout()
        plt.show()
        return

    if ax is None:
        fig, ax = plt.subplots(figsize=(size1, size2))

    p = np.sort(p)
    n = p.size
    running_mean = np.cumsum(p) / np.arange(1, n + 1)
    ax.plot(np.arange(1, n + 1) / n, running_mean, linewidth=2, color=_c1)
    ax.plot(
        [0, 1],
        [0, 0.5],
        "--",
        linewidth=2,
        label="Uniform expectation",
        color=_c2,
    )
    ax.set_xlabel("Fraction of tests", fontsize=fs1)
    ax.set_ylabel("Running mean of p-values", fontsize=fs1)
    ax.legend(fontsize=11)
    _style_axes(ax, "Running mean plot")
    plt.tight_layout()
    plt.show()


"""
def plot_pval_running_mean(pvals, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(size1, size2))

    pvals = np.sort(pvals)
    n = len(pvals)
    running_mean = np.cumsum(pvals) / np.arange(1, n + 1)
    ax.plot(np.arange(1, n + 1) / n, running_mean, color=_c1, linewidth=2)
    ax.plot(
        [0, 1],
        [0, 0.5],
        "--",
        linewidth=2,
        label="Uniform expectation",
        color=_c2,
    )
    ax.set_xlabel("Fraction of tests", fontsize=fs1)
    ax.set_ylabel("Running mean of p-values", fontsize=fs1)
    ax.legend(fontsize=11)
    _style_axes(ax, "Running mean plot")
    plt.tight_layout()
    plt.show()
"""


# ---- 8. Z-score distribution ----
def plot_pval_zdist(pvals, ax=None):
    # Use tiny positive eps so z = norm.isf(p/2) is finite
    p = _clean_pvals(pvals, eps=np.finfo(float).tiny)
    if ax is None:
        fig, ax = plt.subplots(figsize=(size1, size2))
    if p.size == 0:
        ax.text(0.5, 0.5, "No finite p-values", ha="center", va="center")
        _style_axes(ax, "Z distribution (two-sided)")
        plt.tight_layout()
        plt.show()
        return

    zscores = norm.isf(p / 2.0)  # absolute z for two-sided p
    zscores = zscores[np.isfinite(zscores)]
    if zscores.size == 0:
        ax.text(0.5, 0.5, "No finite z-scores", ha="center", va="center")
        _style_axes(ax, "Z distribution (two-sided)")
        plt.tight_layout()
        plt.show()
        return

    ax.hist(zscores, bins=30, density=True, alpha=0.7, color=_c1, edgecolor="k")

    xx = np.linspace(0.0, float(np.max(zscores)), 200)
    ax.plot(
        xx,
        norm.pdf(xx, 0.0, 1.0) * 2.0,
        linewidth=2,
        color=_c2,
        label="Half-N(0,1)",
    )
    ax.set_xlabel("abs(z)", fontsize=fs1)
    ax.set_ylabel("Density", fontsize=fs1)
    ax.legend(fontsize=11)
    _style_axes(ax, "Z distribution (two-sided)")
    plt.tight_layout()
    plt.show()


"""
def plot_pval_zdist(pvals, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(size1, size2))

    zscores = norm.isf(pvals / 2)  # two-sided
    ax.hist(zscores, bins=30, density=True, alpha=0.7, color=_c1, edgecolor="k")
    xx = np.linspace(0, max(zscores), 200)
    ax.plot(
        xx,
        norm.pdf(xx, 0, 1) * 2,
        "--",
        linewidth=2,
        label="|N(0,1)|",
        color=_c2,
    )
    ax.set_xlabel("Z-score (|Φ⁻¹(1-p/2)|)", fontsize=fs1)
    ax.set_ylabel("Density", fontsize=fs1)
    ax.legend(fontsize=11)
    _style_axes(ax, "Z-score distribution")
    plt.tight_layout()
    plt.show()
"""


# ---- 9. Calibration curve ----
def plot_pval_calibration(pvals, n_bins=10, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    pvals = np.asarray(pvals)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    obs_props = [(pvals <= t).mean() for t in bin_edges[1:]]
    ax.plot(
        bin_edges[1:], obs_props, "o-", color=_c1, linewidth=2, label="Observed"
    )
    ax.plot([0, 1], [0, 1], "--", linewidth=2, label="Uniform", color=_c2)
    ax.set_xlabel("Nominal cutoff", fontsize=fs1)
    ax.set_ylabel("Proportion ≤ cutoff", fontsize=fs1)
    ax.legend(fontsize=11)
    _style_axes(ax, "Calibration curve")
    plt.tight_layout()
    plt.show()


def qq_plot_log_p(pvalues, title):
    """
    Generate a Q-Q plot of -log10(p-values) vs expected uniform -log10(p-values).

    Parameters:
    - pvalues: list or array of p-values
    - title: title of the plot
    """
    pvalues = np.asarray(pvalues)
    pvalues = pvalues[np.isfinite(pvalues) & (pvalues > 0)]

    n = len(pvalues)
    if n == 0:
        raise ValueError("No valid p-values to plot.")

    # Sort observed and expected p-values
    observed = -np.log10(np.sort(pvalues))
    expected = -np.log10(np.linspace(1 / (n + 1), n / (n + 1), n))

    # Plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(
        expected,
        observed,
        s=16,
        alpha=0.7,
        edgecolor="dodgerblue",
        linewidth=0.5,
    )

    # Identity line
    max_val = max(np.max(expected), np.max(observed)) * 1.05
    ax.plot(
        [0, max_val],
        [0, max_val],
        linestyle="dotted",
        color="gray",
        linewidth=1.5,
    )

    # Axes styling
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)
    ax.tick_params(width=1.3, labelsize=12)

    ax.set_xlabel("Expected -log10(p)", fontsize=13)
    ax.set_ylabel("Observed -log10(p)", fontsize=13)
    ax.set_title(title, fontsize=14, weight="bold")

    plt.tight_layout()
    plt.show()


def pvalue_histogram(pvalues, title, bins=40, log_y=False):
    """
    Plot a histogram of p-values.

    Parameters:
    - pvalues: list or array of p-values
    - title: str, title of the plot
    - bins: int or sequence, number of histogram bins (default: 40)
    - log_y: bool, whether to use a log scale for the y-axis (default: False)
    """
    pvalues = np.asarray(pvalues)
    pvalues = pvalues[np.isfinite(pvalues) & (pvalues >= 0) & (pvalues <= 1)]

    if len(pvalues) == 0:
        raise ValueError("No valid p-values to plot.")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(pvalues, bins=bins, color="#4C72B0", edgecolor="black", alpha=0.85)

    # Axes styling
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)
    ax.tick_params(width=1.3, labelsize=12)

    ax.set_xlabel("P-value", fontsize=13)
    ax.set_ylabel("Frequency", fontsize=13)
    ax.set_title(title, fontsize=14, weight="bold")

    if log_y:
        ax.set_yscale("log")
        ax.set_ylabel("Frequency (log)", fontsize=13)

    plt.tight_layout()
    plt.show()


def plot_pval_diagnostics(
    pvalues,
    title_left="Q-Q Plot",
    title_right="P-Value Histogram",
    bins=40,
    log_y=False,
):
    """
    Create a side-by-side figure with:
    - Left: Q-Q plot of -log10(p-values).
    - Right: Histogram of raw p-values.

    Parameters
    ----------
    pvalues : array-like
        List or array of p-values.
    title_left : str, default="Q-Q Plot"
        Title for the Q-Q plot.
    title_right : str, default="P-Value Histogram"
        Title for the histogram.
    bins : int, default=40
        Number of histogram bins.
    log_y : bool, default=False
        Whether to use a log scale on the histogram's y-axis.
    """
    pvalues = np.asarray(pvalues)
    pvalues = pvalues[np.isfinite(pvalues) & (pvalues > 0) & (pvalues <= 1)]
    if len(pvalues) == 0:
        raise ValueError("No valid p-values to plot.")

    # Setup figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- Left: Q-Q Plot ---
    n = len(pvalues)
    observed = -np.log10(np.sort(pvalues))
    expected = -np.log10(np.linspace(1 / (n + 1), n / (n + 1), n))

    axes[0].scatter(
        expected,
        observed,
        s=16,
        alpha=0.7,
        edgecolor="dodgerblue",
        linewidth=0.5,
    )
    max_val = max(np.max(expected), np.max(observed)) * 1.05
    axes[0].plot(
        [0, max_val],
        [0, max_val],
        linestyle="dotted",
        color="gray",
        linewidth=1.5,
    )

    axes[0].set_xlabel("Expected -log10(p)", fontsize=13)
    axes[0].set_ylabel("Observed -log10(p)", fontsize=13)
    axes[0].set_title(title_left, fontsize=14, weight="bold")

    for spine in ["top", "right"]:
        axes[0].spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        axes[0].spines[spine].set_linewidth(1.5)
    axes[0].tick_params(width=1.3, labelsize=12)

    # --- Right: Histogram ---
    axes[1].hist(
        pvalues, bins=bins, color="#4C72B0", edgecolor="black", alpha=0.85
    )
    axes[1].set_xlabel("P-value", fontsize=13)
    axes[1].set_ylabel("Frequency", fontsize=13)
    axes[1].set_title(title_right, fontsize=14, weight="bold")

    if log_y:
        axes[1].set_yscale("log")
        axes[1].set_ylabel("Frequency (log)", fontsize=13)

    for spine in ["top", "right"]:
        axes[1].spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        axes[1].spines[spine].set_linewidth(1.5)
    axes[1].tick_params(width=1.3, labelsize=12)

    plt.tight_layout()
    plt.show()


def plot_pvalue_diagnostics_grid(pvals, n_bins=10, sname=None):
    """
    Produce a 3x3 grid of all p-value diagnostic plots by calling the
    individual plotting functions defined earlier.
    """
    fig, axes = plt.subplots(3, 3, figsize=(16, 14))
    axes = axes.flatten()

    # Temporarily override plt.show so individual functions draw into provided axes
    orig_show = plt.show
    plt.show = (
        lambda *args, **kwargs: None
    )  # suppress automatic figure creation

    try:
        # Call each diagnostic, passing the axis explicitly
        plot_pval_histogram(pvals, ax=axes[0])
        plot_pval_qq(pvals, ax=axes[1])
        plot_pval_ecdf(pvals, ax=axes[2])
        plot_pval_cumdev(pvals, ax=axes[3])
        plot_pval_obs_vs_exp(pvals, ax=axes[4])
        plot_pval_survival(pvals, ax=axes[5])
        plot_pval_running_mean(pvals, ax=axes[6])
        plot_pval_zdist(pvals, ax=axes[7])
        plot_pval_calibration(pvals, n_bins=n_bins, ax=axes[8])
    finally:
        plt.show = orig_show  # restore original

    plt.tight_layout()
    if sname is not None:
        plt.savefig(sname)
    plt.show()
