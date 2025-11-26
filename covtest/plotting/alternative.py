import matplotlib.pyplot as plt
import numpy as np


def _style_axes(ax, title):
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)


def plot_power_curve(pvals, params, alpha=0.05):
    """
    pvals: array (n_reps, n_params)
    params: array of parameter values (len = n_params)
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
    pvals: array (n_reps, len(param1), len(param2))
    param1: array for x-axis
    param2: array for y-axis
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
    pvals: array (n_reps, n_params)
    params: array of parameter values
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
    pvals: array (n_reps, n_params)
    params: array of parameter values
    kind: "box" or "violin"
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
    pvals: array (n_reps, n_params)
    params: array of parameter values
    alphas: list of significance levels
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
