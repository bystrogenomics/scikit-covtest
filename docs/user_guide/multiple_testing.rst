.. _multiple-testing:

Multiple Testing Correction
===========================

When performing multiple statistical tests simultaneously, the probability of encountering at least one false positive (Type I error) increases with the number of tests. To control this error rate, ``scikit-covtest`` provides a variety of multiplicity correction procedures divided into two categories:

1. **Family-Wise Error Rate (FWER) control**: Controls the probability of making at least one false rejection (Type I error).
2. **False Discovery Rate (FDR) control**: Controls the expected proportion of false discoveries (false rejections) among all rejected hypotheses.

All procedures are implemented in the :mod:`covtest.multiplicity` submodule.

Family-Wise Error Rate (FWER)
-----------------------------

FWER control is recommended when false positives are highly critical. The following methods are implemented in :mod:`covtest.multiplicity.fwer`:

- **Bonferroni**: The classic single-step correction. Highly conservative but distribution-free.
- **Holm**: A step-down procedure that is more powerful than Bonferroni while making the same distribution-free assumptions.
- **Hochberg**: A step-up procedure that is more powerful than Holm's but requires the p-values to satisfy certain positive dependence conditions (Simes' inequality).
- **Hommel**: A more sophisticated step-up procedure that is slightly more powerful than Hochberg but also computationally intensive.
- **Romano-Wolf max-T**: A bootstrap-based resampling procedure that accounts for the dependence structure of the test statistics, providing significant power gains in correlated settings.

False Discovery Rate (FDR)
--------------------------

FDR control is recommended for large-scale screening tasks where some false positives are acceptable in exchange for a significant increase in statistical power. The following methods are implemented in :mod:`covtest.multiplicity.fdr`:

- **Benjamini-Hochberg (BH)**: The standard step-up procedure that controls FDR under independent or positively dependent p-values.
- **Benjamini-Yekutieli (BY)**: A conservative modification of BH that guarantees FDR control under arbitrary dependence structures.
- **Storey's q-values**: An adaptive procedure that estimates the proportion of true null hypotheses (:math:`\pi_0`) to achieve higher power than BH.
- **Benjamini-Liu (BL)**: A step-down procedure designed to control FDR.
- **BLAROQ**: An adaptive step-up procedure for controlling FDR under dependency.
- **Weighted BH**: A version of the Benjamini-Hochberg procedure where tests can be assigned prior weights based on significance or relevance.

Usage Example
-------------

Below is a complete python example showing how to apply multiplicity corrections to a set of p-values:

.. code-block:: python

   import numpy as np
   from covtest.multiplicity import fwer, fdr

   # Generate hypothetical p-values (some signal, some noise)
   rng = np.random.default_rng(42)
   pvals = np.concatenate([
       rng.uniform(0, 0.001, size=10),
       rng.uniform(0.01, 0.99, size=90)
   ])

   # 1. Apply FWER control (Holm's method)
   holm_res = fwer.holm(pvals, alpha=0.05)
   print("Holm Rejected count:", np.sum(holm_res['reject']))

   # 2. Apply FDR control (Benjamini-Hochberg)
   bh_res = fdr.benjamini_hochberg(pvals, alpha=0.05)
   print("BH Rejected count:", np.sum(bh_res['reject']))

   # 3. Apply Storey's q-values (Adaptive FDR)
   storey_res = fdr.storey_qvalues(pvals, alpha=0.05)
   print("Storey's q-values Rejected count:", np.sum(storey_res['reject']))
