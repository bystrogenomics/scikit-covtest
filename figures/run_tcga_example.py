"""Reproduce the TCGA multiple-testing panels from the paper.

Degenerate five-gene blocks are excluded before multiplicity correction and
recorded in the output.  This is necessary because the Srivastava (2007)
statistic is undefined for a block whose pooled covariance is identically zero.
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

from covtest.datasets import load_tcga
from covtest.methods.hypothesis_two_sample import srivastava_two_sample_2007
from covtest.multiplicity.fdr import benjamini_hochberg, benjamini_yekutieli
from covtest.multiplicity.fwer import bonferroni


COLORS = {"pvalues": "#5B8DB8", "bh": "#78B7A5", "by": "#9C8AC7"}
MAX_LOG_VALUE = 20


def style_axis(ax):
    """Apply the publication style shared by the paper figures."""
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(1.5)
    ax.spines["left"].set_linewidth(1.5)
    ax.tick_params(axis="both", direction="out", length=4.5, width=1.1)


def capped_negative_log10(values):
    """Transform p-values while keeping underflow values visible but finite."""
    return np.minimum(-np.log10(np.maximum(values, 10.0 ** -MAX_LOG_VALUE)), MAX_LOG_VALUE)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block-size", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    return parser.parse_args()


def main():
    args = parse_args()
    X_raw, y = load_tcga()
    X = X_raw[:, 1:].astype(float)  # Drop sample identifier.
    X_brca = StandardScaler().fit_transform(X[y == "BRCA"])
    X_luad = StandardScaler().fit_transform(X[y == "LUAD"])
    n_groups = X.shape[1] // args.block_size
    pvalues = np.full(n_groups, np.nan)
    excluded = []
    for i in range(n_groups):
        idx = args.block_size * i + np.arange(args.block_size)
        try:
            pvalues[i] = srivastava_two_sample_2007(X_brca[:, idx], X_luad[:, idx])["p_value"]
        except ValueError as error:
            excluded.append((i, str(error)))

    valid = np.isfinite(pvalues)
    tested_pvalues = pvalues[valid]
    bh = benjamini_hochberg(tested_pvalues)
    by = benjamini_yekutieli(tested_pvalues)
    bonf = bonferroni(tested_pvalues)
    q_bh = np.full(n_groups, np.nan)
    q_by = np.full(n_groups, np.nan)
    q_bh[valid], q_by[valid] = bh["qvals"], by["qvals"]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2), constrained_layout=True)
    fig.set_constrained_layout_pads(wspace=0.03)
    groups = np.arange(n_groups)
    p_log = capped_negative_log10(tested_pvalues)
    q_bh_log = capped_negative_log10(q_bh[valid])
    q_by_log = capped_negative_log10(q_by[valid])
    bonf_threshold = -np.log10(0.05 / tested_pvalues.size)
    bonf_rejected = tested_pvalues <= 0.05 / tested_pvalues.size
    axes[0].scatter(groups[valid], p_log, s=10, color="#B7C2D0", alpha=0.72, linewidths=0)
    axes[0].scatter(groups[valid][bonf_rejected], p_log[bonf_rejected], s=22, color=COLORS["pvalues"], label=f"Bonferroni rejections ({bonf_rejected.sum()})", zorder=3)
    axes[0].axhline(bonf_threshold, color="#6F6F6F", linestyle=(0, (4, 3)), linewidth=1.4, label="Bonferroni threshold")
    axes[0].set_xlabel("Covariate group", fontsize=18)
    axes[0].set_ylabel("-log10(p-value)", fontsize=18)
    axes[0].set_title("Block-level p-values", fontsize=24, pad=12)
    axes[0].legend(frameon=False, fontsize=10, loc="upper right")
    style_axis(axes[0])

    axes[1].scatter(groups[valid], q_bh_log, s=11, color=COLORS["bh"], alpha=0.78, linewidths=0, label=f"BH ({bh['rejected'].sum()} rejections)")
    axes[1].scatter(groups[valid], q_by_log, s=11, color=COLORS["by"], alpha=0.72, linewidths=0, label=f"BY ({by['rejected'].sum()} rejections)")
    axes[1].axhline(-np.log10(0.05), color="#6F6F6F", linestyle=(0, (4, 3)), linewidth=1.4, label=r"$q = 0.05$")
    axes[1].set_xlabel("Covariate group", fontsize=18)
    axes[1].set_ylabel("-log10(adjusted p-value)", fontsize=18)
    axes[1].set_title("FDR-adjusted p-values",fontsize=24)
    axes[1].legend(frameon=False, fontsize=10, loc="upper left")
    style_axis(axes[1])

    for ax in axes:
        ax.set_xlim(0, n_groups - 1)
        ax.margins(x=0)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_dir / "figure_tcga.pdf")
    fig.savefig(args.output_dir / "figure_tcga.png", dpi=300)
    np.savez(args.output_dir / "tcga_example_results.npz", pvalues=pvalues, q_bh=q_bh, q_by=q_by, valid=valid)
    print(f"Tested {tested_pvalues.size}/{n_groups} blocks; excluded {len(excluded)} degenerate blocks.")
    print(f"Bonferroni={bonf['rejected'].sum()}, BH={bh['rejected'].sum()}, BY={by['rejected'].sum()}")


if __name__ == "__main__":
    main()
