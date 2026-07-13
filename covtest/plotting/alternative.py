import matplotlib.pyplot as plt
import numpy as np


def _style_axes(ax, title):
    """Apply consistent styling to matplotlib axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes object to style.
    title : str
        Title for the plot.
    """
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)


def plot_power_curve(pvals, params, alpha=0.05):
    """
    Plot statistical power as a function of a parameter.

    Creates a line plot showing how power (rejection rate) changes across
    different parameter values, with a reference line at 80% power.

    Parameters
    ----------
    pvals : array-like of shape (n_reps, n_params)
        P-values from simulation replicates. Each row is one replicate,
        each column corresponds to a parameter value.

    params : array-like of shape (n_params,)
        Parameter values for the x-axis.

    alpha : float, default=0.05
        Significance level for determining rejections.

    Returns
    -------
    None
        Displays the plot using matplotlib.

    Notes
    -----
    Power is computed as the proportion of p-values below alpha for each
    parameter value. The plot includes a horizontal reference line at 0.8
    (80% power), which is a common target in power analysis.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> # Simulate p-values with increasing power
    >>> params = np.linspace(0, 1, 10)
    >>> pvals = np.random.beta(1, 1 + 5*(1-params), size=(100, 10))
    >>> plot_power_curve(pvals, params, alpha=0.05)
    """
    power = (pvals < alpha).mean(axis=0)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(params, power, "o-", color="steelblue", linewidth=2)
    ax.axhline(0.8, color="red", linestyle="--", linewidth=2, label="80% power")
    ax.set_xlabel("Parameter", fontsize=12)
    ax.set_ylabel("Power (rejection rate)", fontsize=12)
    ax.legend(fontsize=11)
    _style_axes(ax, f"Power curve (α={alpha})")
    plt.tight_layout()
    plt.show()


def plot_power_heatmap(pvals, param1, param2, alpha=0.05):
    """
    Plot statistical power as a 2D heatmap over two parameters.

    Creates a heatmap showing how power varies across combinations of two
    different parameters, useful for visualizing power surfaces.

    Parameters
    ----------
    pvals : array-like of shape (n_reps, len(param1), len(param2))
        P-values from simulation replicates. First dimension is replicates,
        second and third dimensions correspond to param1 and param2 grids.

    param1 : array-like of shape (n_param1,)
        Parameter values for the y-axis.

    param2 : array-like of shape (n_param2,)
        Parameter values for the x-axis.

    alpha : float, default=0.05
        Significance level for determining rejections.

    Returns
    -------
    None
        Displays the plot using matplotlib.

    Notes
    -----
    Power is computed as the proportion of p-values below alpha for each
    (param1, param2) combination. The heatmap uses the viridis colormap
    with a colorbar indicating power levels.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> param1 = np.linspace(0, 1, 5)
    >>> param2 = np.linspace(0, 1, 5)
    >>> # Simulate p-values with power depending on both parameters
    >>> pvals = np.random.beta(1, 1 + 3*(1-param1[:, None]) + 2*(1-param2),
    ...                        size=(100, 5, 5))
    >>> plot_power_heatmap(pvals, param1, param2, alpha=0.05)
    """
    power = (pvals < alpha).mean(axis=0)  # shape (len(param1), len(param2))
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(
        power,
        origin="lower",
        aspect="auto",
        extent=[param2.min(), param2.max(), param1.min(), param1.max()],
        cmap="viridis",
    )
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Power", fontsize=12)
    ax.set_xlabel("Parameter 2", fontsize=12)
    ax.set_ylabel("Parameter 1", fontsize=12)
    _style_axes(ax, f"Power heatmap (α={alpha})")
    plt.tight_layout()
    plt.show()


def plot_mean_pvalue(pvals, params):
    """
    Plot mean p-value trajectory across parameter values.

    Creates a line plot showing how the mean p-value changes across
    different parameter values, with a reference line at α=0.05.

    Parameters
    ----------
    pvals : array-like of shape (n_reps, n_params)
        P-values from simulation replicates. Each row is one replicate,
        each column corresponds to a parameter value.

    params : array-like of shape (n_params,)
        Parameter values for the x-axis.

    Returns
    -------
    None
        Displays the plot using matplotlib.

    Notes
    -----
    The mean p-value provides a summary of the central tendency of the
    p-value distribution at each parameter value. Under the null hypothesis,
    mean p-values should be around 0.5. Decreasing mean p-values indicate
    increasing evidence against the null.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> params = np.linspace(0, 1, 10)
    >>> pvals = np.random.beta(1, 1 + 3*(1-params), size=(100, 10))
    >>> plot_mean_pvalue(pvals, params)
    """
    mean_p = pvals.mean(axis=0)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(params, mean_p, "o-", color="steelblue", linewidth=2)
    ax.axhline(0.05, color="red", linestyle="--", linewidth=2, label="α=0.05")
    ax.set_xlabel("Parameter", fontsize=12)
    ax.set_ylabel("Mean p-value", fontsize=12)
    ax.legend(fontsize=11)
    _style_axes(ax, "Mean p-value trajectory")
    plt.tight_layout()
    plt.show()


def plot_pvalue_distributions(pvals, params, kind="box"):
    """
    Plot p-value distributions across parameter values.

    Creates box plots or violin plots showing the distribution of p-values
    at each parameter value, useful for visualizing variability.

    Parameters
    ----------
    pvals : array-like of shape (n_reps, n_params)
        P-values from simulation replicates. Each row is one replicate,
        each column corresponds to a parameter value.

    params : array-like of shape (n_params,)
        Parameter values for the x-axis.

    kind : {'box', 'violin'}, default='box'
        Type of distribution plot to create.

    Returns
    -------
    None
        Displays the plot using matplotlib.

    Notes
    -----
    Box plots show the median, quartiles, and outliers of the p-value
    distribution at each parameter value. Violin plots additionally show
    the kernel density estimate of the distribution.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> params = np.array([0.0, 0.5, 1.0])
    >>> pvals = np.random.beta(1, 1 + 3*(1-params), size=(100, 3))
    >>> plot_pvalue_distributions(pvals, params, kind='box')
    >>> plot_pvalue_distributions(pvals, params, kind='violin')
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    if kind == "box":
        ax.boxplot(
            [pvals[:, j] for j in range(len(params))],
            positions=params,
            widths=0.05 * np.ptp(params),
        )
    elif kind == "violin":
        parts = ax.violinplot(
            [pvals[:, j] for j in range(len(params))],
            positions=params,
            widths=0.05 * np.ptp(params),
            showmeans=True,
            showextrema=False,
        )
        for pc in parts["bodies"]:
            pc.set_facecolor("steelblue")
            pc.set_edgecolor("black")
            pc.set_alpha(0.6)
    ax.set_xlabel("Parameter", fontsize=12)
    ax.set_ylabel("p-value", fontsize=12)
    ax.set_ylim(0, 1)
    _style_axes(ax, f"P-value distributions ({kind})")
    plt.tight_layout()
    plt.show()


def plot_power_curves_multi_alpha(pvals, params, alphas=[0.1, 0.05, 0.01]):
    """
    Plot power curves at multiple significance levels.

    Creates overlaid power curves showing how power changes across parameter
    values for different significance levels, useful for comparing sensitivity.

    Parameters
    ----------
    pvals : array-like of shape (n_reps, n_params)
        P-values from simulation replicates. Each row is one replicate,
        each column corresponds to a parameter value.

    params : array-like of shape (n_params,)
        Parameter values for the x-axis.

    alphas : list of float, default=[0.1, 0.05, 0.01]
        Significance levels to plot.

    Returns
    -------
    None
        Displays the plot using matplotlib.

    Notes
    -----
    Each curve shows the power (rejection rate) at a different significance
    level. Higher alpha values result in higher power but also higher Type I
    error rates. The plot includes a reference line at 80% power.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> params = np.linspace(0, 1, 10)
    >>> pvals = np.random.beta(1, 1 + 5*(1-params), size=(100, 10))
    >>> plot_power_curves_multi_alpha(pvals, params, alphas=[0.1, 0.05, 0.01])
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    for alpha in alphas:
        power = (pvals < alpha).mean(axis=0)
        ax.plot(params, power, "o-", linewidth=2, label=f"α={alpha}")
    ax.axhline(
        0.8, color="gray", linestyle="--", linewidth=2, label="80% power"
    )
    ax.set_xlabel("Parameter", fontsize=12)
    ax.set_ylabel("Power", fontsize=12)
    ax.legend(fontsize=11)
    _style_axes(ax, "Power curves at multiple α")
    plt.tight_layout()
    plt.show()
