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
    schott2007,
    srivastava_two_sample_2007,
    srivastava_yanagihara_two_sample,
)


METHODS = [
    ("Srivastava (2007)", srivastava_two_sample_2007),
    ("Schott (2007)", schott2007),
    ("Srivastava-Yanagihara (2014)", srivastava_yanagihara_two_sample),
]


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

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    axes[0].hist(X0.ravel(), bins=100, density=True, alpha=0.6, label="MNIST: 0")
    axes[0].hist(X1.ravel(), bins=100, density=True, alpha=0.6, label="MNIST: 1")
    axes[0].set(xlabel="Standardized input", ylabel="Density", title="Data distributions")
    axes[0].legend()
    for j, (label, _) in enumerate(METHODS):
        axes[1].hist(null[:, j], bins=30, density=True, histtype="step", label=label)
        axes[2].hist(-np.log10(np.maximum(alternative[:, j], np.finfo(float).tiny)), bins=30, density=True, histtype="step", label=label)
    axes[1].set(xlabel="P-value", ylabel="Density", title="Null p-values", xlim=(0, 1))
    axes[2].set(xlabel="-log10(P-value)", ylabel="Density", title="Alternative p-values")
    axes[1].legend(fontsize=8)
    axes[2].legend(fontsize=8)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_dir / "figure_mnist.pdf")
    fig.savefig(args.output_dir / "figure_mnist.png", dpi=300)
    np.savez(args.output_dir / "mnist_example_results.npz", null=null, alternative=alternative)


if __name__ == "__main__":
    main()
