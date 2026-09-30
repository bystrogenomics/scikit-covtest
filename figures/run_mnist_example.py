"""Reproduce the MNIST panels used in the paper's worked example.

Run with ``python figures/run_mnist_example.py`` after installing the dataset
extra.  Defaults are deliberately explicit so that a manuscript can cite the
same design: N=200 observations per group, 300 replications, and seed 0.
"""

import argparse
import sys
from pathlib import Path

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/scikit-covtest-matplotlib")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

# Allow ``python figures/<script>.py`` from a source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from covtest.datasets import load_mnist
from covtest.methods.hypothesis_two_sample import (
    srivastava_two_sample_2007,
    srivastava_yanagihara_two_sample,
)


METHODS = [
    ("Srivastava (2007)", srivastava_two_sample_2007),
    ("Srivastava-Yanagihara (2014)", srivastava_yanagihara_two_sample),
]

COLORS = ["#5B8DB8", "#78B7A5", "#9C8AC7"]
NULL_COLOR = "#9C8AC7"
ALT_COLOR = "#E39B63"


def style_axis(ax):
    """Apply the publication style shared by the paper figures."""
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(1.5)
    ax.spines["left"].set_linewidth(1.5)
    ax.tick_params(
        axis="both", direction="out", length=4.5, width=1.1, labelsize=14
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=200, help="Samples per group.")
    parser.add_argument("--n-rep", type=int, default=300, help="Replications.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    return parser.parse_args()


def pvalue_samples(X0, X1, n, n_rep, rng):
    if 2 * n > len(X0) or n > len(X1):
        raise ValueError("Requested sample size exceeds the available MNIST data.")
    null = np.empty((n_rep, len(METHODS)))
    alternative = np.empty_like(null)
    for i in range(n_rep):
        idx0 = rng.choice(len(X0), size=2 * n, replace=False)
        idx1 = rng.choice(len(X1), size=n, replace=False)
        for j, (_, method) in enumerate(METHODS):
            null[i, j] = method(X0[idx0[:n]], X0[idx0[n:]])["p_value"]
            alternative[i, j] = method(X0[idx0[:n]], X1[idx1])["p_value"]
    return null, alternative


def main():
    args = parse_args()
    X_train, y_train = load_mnist(split="train")
    X_test, y_test = load_mnist(split="test")
    X = np.concatenate((X_train, X_test))
    y = np.concatenate((y_train, y_test))
    X0 = StandardScaler().fit_transform(X[y == 0])
    X1 = StandardScaler().fit_transform(X[y == 1])
    X0[X0 > 10] = 10
    X1[X1 > 10] = 10
    null, alternative = pvalue_samples(
        X0, X1, args.n, args.n_rep, np.random.default_rng(args.seed)
    )

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), constrained_layout=True)
    fig.set_constrained_layout_pads(wspace=0.05)
    axes[0].hist(
        X0.ravel(), bins=100, density=True, alpha=0.70, color=COLORS[0],
        label="Digit 0",
    )
    axes[0].hist(
        X1.ravel(), bins=100, density=True, alpha=0.60, color=ALT_COLOR,
        label="Digit 1",
    )
    axes[0].set_xlim(-3, 5)
    axes[0].set(
        xlabel="Standardized pixel intensity", ylabel="Density",
        title="MNIST input distributions",
    )
    axes[0].set_xlabel("Standardized pixel intensity", fontsize=18)
    axes[0].set_ylabel("Density", fontsize=18)
    axes[0].set_title("MNIST input distributions", fontsize=20, pad=12)
    axes[0].legend(frameon=False, fontsize=11)
    style_axis(axes[0])

    for j, (label, _) in enumerate(METHODS):
        observed = np.sort(null[:, j])
        expected = np.arange(1, len(observed) + 1) / len(observed)
        axes[1].plot(expected, observed, color=COLORS[j], linewidth=2.4, label=label)
    axes[1].plot([0, 1], [0, 1], color="#6F6F6F", linestyle=(0, (4, 3)), linewidth=1.3)
    axes[1].set(
        xlabel="Expected p-value", ylabel="Observed p-value",
        title="Null calibration", xlim=(0, 1), ylim=(0, 1),
    )
    axes[1].set_xlabel("Expected p-value", fontsize=18)
    axes[1].set_ylabel("Observed p-value", fontsize=18)
    axes[1].set_title("Null calibration", fontsize=20, pad=12)
    axes[1].set_aspect("equal", adjustable="box")
    axes[1].legend(frameon=False, fontsize=10, loc="upper left")
    style_axis(axes[1])

    x = np.arange(len(METHODS))
    width = 0.36
    null_rate = np.mean(null < 0.05, axis=0)
    power = np.mean(alternative < 0.05, axis=0)
    axes[2].bar(x - width / 2, null_rate, width, color=NULL_COLOR, label="Null")
    axes[2].bar(x + width / 2, power, width, color=ALT_COLOR, label="Digit 0 vs 1")
    axes[2].axhline(0.05, color="#6F6F6F", linestyle=(0, (4, 3)), linewidth=1.3)
    axes[2].text(
        x[-1] + 0.37, 0.07, r"$\alpha = 0.05$", color="#5A5A5A",
        ha="right", fontsize=11,
    )
    axes[2].set(
        xticks=x, xticklabels=["Srivastava", "S.-Y."],
        xlabel="Test", ylabel="Rejection rate", title="Null rejection and power",
        ylim=(0, 1.05),
    )
    axes[2].set_xlabel("Test", fontsize=18)
    axes[2].set_ylabel("Rejection rate", fontsize=18)
    axes[2].set_title("Null rejection and power", fontsize=20, pad=12)
    axes[2].legend(frameon=False, fontsize=11, loc="upper left")
    style_axis(axes[2])
    for value, position in zip(null_rate, x - width / 2):
        axes[2].text(position, min(value + 0.03, 1.02), f"{value:.2f}", ha="center", fontsize=9)
    for value, position in zip(power, x + width / 2):
        axes[2].text(position, min(value + 0.03, 1.02), f"{value:.2f}", ha="center", fontsize=9)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_dir / "figure_mnist.pdf")
    fig.savefig(args.output_dir / "figure_mnist.png", dpi=300)
    np.savez(args.output_dir / "mnist_example_results.npz", null=null, alternative=alternative)


if __name__ == "__main__":
    main()
