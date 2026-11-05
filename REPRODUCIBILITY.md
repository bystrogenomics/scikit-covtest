# Paper reproducibility and revision notes

Run the following commands from the repository root with Python 3.10 or later.
Record the Git revision used to make any manuscript figure:

```bash
git rev-parse HEAD
```

This repository contains one script per paper figure:

| Paper figure | Script | Output |
| --- | --- | --- |
| Figure 4 | `figures/run_figure_simulations.py` | `figures/figure_simulations.pdf` and `figures/figure_simulations.png` |
| Figure 5 | `figures/run_mnist_example.py` | `figures/figure_mnist.pdf` and `figures/figure_mnist.png`, plus `figures/mnist_example_results.npz` |
| Figure 6 | `figures/run_tcga_example.py` | `figures/figure_tcga.pdf` and `figures/figure_tcga.png`, `figures/tcga_pathway_results.csv`, and `figures/tcga_pathway_provenance.txt`; with `--calibration-reps > 0`, also `figures/figure_tcga_null_calibration.pdf`, `figures/figure_tcga_null_calibration.png`, and `figures/tcga_null_calibration.csv` |

## Install and run

Install the optional dataset dependency before running the MNIST or TCGA
examples:

```bash
python -m pip install -e ".[datasets]"
python figures/run_figure_simulations.py
python figures/run_mnist_example.py --n 200 --n-rep 300 --seed 0
python figures/run_tcga_example.py --min-genes 15 --max-genes 100 \
    --calibration-reps 1000 --seed 42
```

MNIST and TCGA are downloaded on first use from the pinned dataset release and
cached locally. The loader verifies the configured SHA-256 checksum before
using each downloaded file. The TCGA script additionally requires the
repository-local `figures/data/tcga_pancan_gene_ids.txt` and
`figures/data/reactome_pathways_2026-06-21.gmt` files. A release or archival
copy must include those inputs; their SHA-256 checksums are recorded in
`figures/tcga_pathway_provenance.txt`.

The MNIST command writes p-values to `figures/mnist_example_results.npz`. Its
defaults are `N=200`, `n_rep=300`, and seed `0`; state these values in the
manuscript whenever those results are reported.

## TCGA worked example

The TCGA workflow is an exploratory software demonstration, not a
confirmatory biological analysis. It drops the sample-ID column, restores
source PanCan gene identifiers, and tests human Reactome pathways containing
15--100 uniquely matched genes. BRCA and LUAD are separately processed with
`StandardScaler`, so the target is standardized within-pathway covariance
(correlation-like) structure rather than raw covariance.

With the supplied repository-local inputs and the exact settings above, all
1,152 eligible
pathways were tested. The results CSV contains the raw analytic `p_value`,
BH and BY adjusted values (`q_bh` and `q_by`), and a Boolean Bonferroni
decision (`bonferroni_rejected`); it does not contain a numeric
Bonferroni-adjusted p-value. Treat these values as conventional nominal
summaries for the worked example, not as validated biological discoveries or
empirically established FWER/FDR guarantees.

The command above also runs a limited null diagnostic: 1,000 Gaussian draws
for each of the 85 observed pathway dimensions, using BRCA- and LUAD-sized
groups, a shared AR(1) correlation matrix with $\rho=0.5$, and the same
within-group scaling and test pipeline. At nominal $\alpha=0.05$, the observed
per-dimension rejection rates range from 0 to 0.001. This shows strong
conservatism in that specified simulation model only; it does not establish
calibration for TCGA data, other covariance structures, non-Gaussian data, or
FWER/FDR control.

The pathway test call can raise `ValueError`, in which case the script skips
that pathway. Before reporting results, compare the printed tested-pathway
count with the number of data rows in `tcga_pathway_results.csv`. The
provenance file records the two input checksums and the pathway-size rule, not
the identities of any skipped pathways.

## Submission checklist

- [x] Regenerate Figure 4 with the current simulation script. The resulting
  Ledoit--Wolf power at $n=50$, 75, and 100 is 0.46, 0.62, and 0.7733 under
  the fixed design.
- [x] Regenerate Figure 5 with `N=200`, `n_rep=300`, and RNG seed 0.
- [x] Regenerate the exploratory Figure 6 pathway result and its AR(1) null
  diagnostic with the command above.
- [ ] Update the complete manuscript source (not merely a rendered PDF):
  insert the regenerated Figures 4--6, state the MNIST settings, and replace
  the retired five-gene-block TCGA text with the exploratory Reactome pathway
  workflow. It should describe 1,152 tested pathways with 15--100 uniquely
  matched genes, standardized within-pathway covariance structure, analytic
  p-values, BH/BY adjusted values, and a Bonferroni rejection indicator. Do
  not present the AR(1) diagnostic as TCGA-wide calibration or FWER/FDR
  validation.
- [ ] Change the software version stated for the examples from 0.1.0 to 0.1.2
  and state the commit/tag used to generate the figures.
- [ ] Replace the claim that a "full notebook" is available with the accurate
  statement that the repository provides the three figure scripts above, or
  add and archive a fully executed notebook before retaining that claim.
- [ ] Update the availability/reproducibility statement to link to a permanent
  tagged release or archival DOI containing these scripts, the required
  pathway inputs, generated artifacts, and the exact manuscript source.

## Statistical safeguards

FDR procedures reject non-finite p-values with `ValueError`; callers must
resolve or explicitly exclude invalid tests rather than allowing NaNs to
silently contaminate adjusted p-values. The Srivastava (2007) two-sample test
also raises a clear `ValueError` for a zero pooled trace-square estimate. The
TCGA pathway script skips a pathway whose test call raises `ValueError`, so
the results CSV row count must be checked against the printed tested-pathway
count before the testing family is reported.
