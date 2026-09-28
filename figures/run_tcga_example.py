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

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    groups = np.arange(n_groups)
    axes[0].scatter(groups[valid], -np.log10(tested_pvalues), s=7)
    axes[0].axhline(-np.log10(0.05 / tested_pvalues.size), color="black", linestyle="--", label="Bonferroni 0.05")
    axes[0].set(xlabel="Covariate group", ylabel="-log10(P-value)", title="P-values: BRCA vs LUAD")
    axes[0].legend()
    axes[1].scatter(groups[valid], -np.log10(np.maximum(q_bh[valid], np.finfo(float).tiny)), s=7, label="BH")
    axes[1].scatter(groups[valid], -np.log10(np.maximum(q_by[valid], np.finfo(float).tiny)), s=7, label="BY")
    axes[1].set(xlabel="Covariate group", ylabel="-log10(Q-value)", title="Q-values: BRCA vs LUAD")
    axes[1].legend()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_dir / "figure_tcga.pdf")
    fig.savefig(args.output_dir / "figure_tcga.png", dpi=300)
    np.savez(args.output_dir / "tcga_example_results.npz", pvalues=pvalues, q_bh=q_bh, q_by=q_by, valid=valid)
    print(f"Tested {tested_pvalues.size}/{n_groups} blocks; excluded {len(excluded)} degenerate blocks.")
    print(f"Bonferroni={bonf['rejected'].sum()}, BH={bh['rejected'].sum()}, BY={by['rejected'].sum()}")


if __name__ == "__main__":
    main()
