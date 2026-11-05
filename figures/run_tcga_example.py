"""Demonstrate pathway-level covariance testing for BRCA versus LUAD.

The expression values remain in their distributed order.  The accompanying
PanCan identifier map restores that order's source identifiers and Reactome
supplies the biological pathway membership.  This is an exploratory worked
example: its analytic p-values and multiplicity adjustments are descriptive
summaries, not validated biological-discovery claims.
"""
import argparse
import csv
import hashlib
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from covtest.datasets import load_tcga
from covtest.methods.hypothesis_two_sample import srivastava_two_sample_2007
from covtest.multiplicity.fdr import benjamini_hochberg, benjamini_yekutieli
from covtest.multiplicity.fwer import bonferroni

HERE = Path(__file__).resolve().parent
GENE_IDS = HERE / "data/tcga_pancan_gene_ids.txt"
PATHWAYS = HERE / "data/reactome_pathways_2026-06-21.gmt"


def read_pathways(n_columns, minimum, maximum):
    identifiers = GENE_IDS.read_text().splitlines()
    if identifiers[0] != "sample" or len(identifiers) != n_columns + 1:
        raise ValueError("TCGA identifier map does not match expression columns")
    index = {}
    for i, gene in enumerate(identifiers[1:]):
        index.setdefault(gene, i)  # Do not test duplicated source identifiers twice.
    pathways = []
    for line in PATHWAYS.read_text().splitlines():
        name, pathway_id, *genes = line.split("\t")
        if not pathway_id.startswith("R-HSA-"):
            continue
        indices = list(dict.fromkeys(index[g] for g in genes if g in index))
        if minimum <= len(indices) <= maximum:
            pathways.append((name, pathway_id, np.asarray(indices)))
    if not pathways:
        raise ValueError("No eligible Reactome pathways")
    return pathways


def test_pathways(brca, luad, pathways):
    rows = []
    for name, pathway_id, indices in pathways:
        try:
            test = srivastava_two_sample_2007(brca[:, indices], luad[:, indices])
        except ValueError:
            continue
        rows.append({"pathway_id": pathway_id, "pathway_name": name,
                     "n_genes": len(indices), "stat": test["stat"],
                     "p_value": test["p_value"]})
    pvalues = np.array([row["p_value"] for row in rows])
    bh, by, bonf = (benjamini_hochberg(pvalues),
                    benjamini_yekutieli(pvalues), bonferroni(pvalues))
    for row, bh_q, by_q, rejected in zip(rows, bh["qvals"], by["qvals"], bonf["rejected"]):
        row.update(q_bh=bh_q, q_by=by_q, bonferroni_rejected=bool(rejected))
    return rows


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def calibrate(pathways, n_brca, n_luad, repetitions, seed, output_dir):
    """Run a finite-sample diagnostic under a structured Gaussian covariance null.

    Each distinct pathway dimension is evaluated, using a shared AR(1)
    correlation matrix in both groups and the same within-group scaling as the
    TCGA analysis. This limited diagnostic examines the analytic reference at
    the dimensions used in the pathway analysis; it does not validate
    calibration for TCGA or FWER/FDR control.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    dimensions = sorted({len(indices) for _, _, indices in pathways})
    records, all_pvalues = [], []
    for dimension in dimensions:
        covariance = 0.5 ** np.abs(np.subtract.outer(np.arange(dimension), np.arange(dimension)))
        pvalues = []
        for _ in range(repetitions):
            brca = rng.multivariate_normal(np.zeros(dimension), covariance, n_brca)
            luad = rng.multivariate_normal(np.zeros(dimension), covariance, n_luad)
            brca = StandardScaler().fit_transform(brca)
            luad = StandardScaler().fit_transform(luad)
            pvalues.append(srivastava_two_sample_2007(brca, luad)["p_value"])
        pvalues = np.asarray(pvalues)
        all_pvalues.extend(pvalues)
        records.append({"dimension": dimension, "n_repetitions": repetitions,
                        "rejection_rate_alpha_0.05": (pvalues <= .05).mean()})
    with (output_dir / "tcga_null_calibration.csv").open("w", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=records[0].keys(), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(records)
    pvalues = np.sort(np.asarray(all_pvalues))
    expected = (np.arange(pvalues.size) + 1) / pvalues.size
    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    ax.plot(expected, pvalues, color="#5B8DB8", lw=1.5)
    ax.plot([0, 1], [0, 1], color="0.4", ls="--")
    ax.set(xlabel="Expected uniform p-value", ylabel="Observed p-value",
           title="TCGA pathway-dimension null calibration", xlim=(0, 1), ylim=(0, 1))
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(output_dir / "figure_tcga_null_calibration.pdf")
    fig.savefig(output_dir / "figure_tcga_null_calibration.png", dpi=300)


def save(rows, output_dir, args):
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "tcga_pathway_results.csv").open("w", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=rows[0].keys(), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: row["p_value"]))
    with (output_dir / "tcga_pathway_provenance.txt").open("w") as file:
        file.write(f"gene_id_map_sha256={digest(GENE_IDS)}\n")
        file.write(f"reactome_gmt_sha256={digest(PATHWAYS)}\n")
        file.write(f"eligible_pathway_size={args.min_genes}-{args.max_genes}\n")
    ordered = sorted(rows, key=lambda row: row["p_value"])
    p = np.array([row["p_value"] for row in ordered])
    q = np.array([row["q_by"] for row in ordered])
    rank = np.arange(1, len(rows) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2), constrained_layout=True)
    axes[0].scatter(rank, -np.log10(np.maximum(p, 1e-20)), s=12, color="#5B8DB8")
    axes[0].axhline(-np.log10(.05 / len(rows)), color="0.4", ls="--")
    axes[0].set(xlabel="Reactome pathway rank", ylabel="-log10(analytic p-value)",
                title="Pathway-level analytic p-values")
    axes[1].scatter(rank, -np.log10(np.maximum(q, 1e-20)), s=12, color="#9C8AC7")
    axes[1].axhline(-np.log10(.05), color="0.4", ls="--")
    axes[1].set(xlabel="Reactome pathway rank", ylabel="-log10(BY-adjusted value)",
                title="BY-adjusted analytic p-values")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(output_dir / "figure_tcga.pdf")
    fig.savefig(output_dir / "figure_tcga.png", dpi=300)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-genes", type=int, default=15)
    parser.add_argument("--max-genes", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=HERE)
    parser.add_argument("--calibration-reps", type=int, default=0,
                        help="null replicates for each distinct pathway dimension")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    raw, labels = load_tcga()
    X = raw[:, 1:].astype(float)
    pathways = read_pathways(X.shape[1], args.min_genes, args.max_genes)
    brca = StandardScaler().fit_transform(X[labels == "BRCA"])
    luad = StandardScaler().fit_transform(X[labels == "LUAD"])
    rows = test_pathways(brca, luad, pathways)
    save(rows, args.output_dir, args)
    if args.calibration_reps:
        calibrate(pathways, brca.shape[0], luad.shape[0],
                  args.calibration_reps, args.seed, args.output_dir)
    print(f"Tested {len(rows)} eligible Reactome pathways (exploratory workflow).")


if __name__ == "__main__":
    main()
