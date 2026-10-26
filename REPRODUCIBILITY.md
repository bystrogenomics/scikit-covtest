# Paper reproducibility and revision notes

This repository contains one script per paper figure:

| Paper figure | Script | Output |
| --- | --- | --- |
| Figure 4 | `figures/run_figure_simulations.py` | `figures/figure_simulations.pdf` and `.png` |
| Figure 5 | `figures/run_mnist_example.py` | `figures/figure_mnist.pdf` and `.png` |
| Figure 6 | `figures/run_tcga_example.py` | `figures/figure_tcga.pdf` and `.png` |

Install the optional dataset dependency first:

```bash
pip install -e ".[datasets]"
python figures/run_figure_simulations.py
python figures/run_mnist_example.py --n 200 --n-rep 300 --seed 0
python figures/run_tcga_example.py
```

The MNIST script writes its p-values to `figures/mnist_example_results.npz`.
Its defaults are `N=200`, `n_rep=300`, and seed `0`; these values must be
stated in the manuscript whenever those results are reported.

The TCGA script drops the sample-ID column, standardizes BRCA and LUAD
separately, forms consecutive blocks of five features, and records invalid
blocks as NaN in its output file. It excludes those blocks before Bonferroni,
BH, or BY correction. With the pinned TCGA dataset, 30 of 4,106 blocks are
degenerate and 4,076 are tested. The resulting counts are Bonferroni 23, BH
41, and BY 26.

## Required manuscript changes

Before resubmission, update the paper source (not merely the rendered PDF):

1. Change the software version for the examples from 0.1.0 to 0.1.2, and
   state the commit/tag used to generate the figures.
2. Regenerate and replace Figure 4 with the output of the current script. In
   particular, the Ledoit-Wolf power at n = 50, 75, and 100 is 0.46, 0.62,
   and 0.7733 with the fixed design.
3. Replace Figures 5 and 6 with the outputs from the new scripts. State the
   MNIST values of `N`, `n_rep`, and the RNG seed in the example text.
4. Revise the TCGA section to say that 30 degenerate blocks were excluded
   before multiplicity correction, that the correction family has 4,076
   valid tests, and that the rejection counts are 23 (Bonferroni), 41 (BH),
   and 26 (BY). Do not report the previous 42 and 27 counts.
5. Replace the claim that a "full notebook" is available with the accurate
   statement that the repository provides the three figure scripts above, or
   add and archive a fully executed notebook before retaining that claim.
6. Update the availability/reproducibility statement to link to a permanent
   tagged release or archival DOI containing these scripts and the exact
   manuscript source.

## Statistical safeguards

FDR procedures now reject non-finite p-values with `ValueError`; callers must
resolve or explicitly exclude invalid tests rather than allowing NaNs to
silently contaminate adjusted p-values. The Srivastava (2007) two-sample test
also raises a clear `ValueError` for a zero pooled trace-square estimate,
which is the degenerate constant-block case encountered in TCGA.
