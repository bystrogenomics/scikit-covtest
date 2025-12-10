.. _covtest-diagnostics:

Diagnostics
===========

This page documents the diagnostic utilities implemented in :mod:`covtest.diagnostics`.
These include tests for multivariate normality and tools for evaluating p-value distributions.

Assumptions and normality tests
-------------------------------

.. currentmodule:: covtest.diagnostics.assumptions

.. automodule:: covtest.diagnostics.assumptions
   :no-members:

.. autosummary::
   :toctree: generated/
   :template: function.rst

   eigen_spectrum
   mardia_tests
   shapiro_francia_w
   royston_test
   hz_test
   condition_and_rank

P-value evaluation
------------------

.. currentmodule:: covtest.diagnostics.evaluate_pvalues

.. automodule:: covtest.diagnostics.evaluate_pvalues
   :no-members:

.. autosummary::
   :toctree: generated/
   :template: function.rst

   analyze_pvalues
